#!/usr/bin/env python3
"""Test whether FP-W4A6 residual directions change across diffusion phases.

The full-precision trajectory supplies a shared x_t.  At selected timesteps,
the same x_t is evaluated in FP and W4A6 modes.  Selected operator outputs are
sampled, and the residual R = Y_FP - Y_W4A6 is analyzed with low-rank output
subspaces.  A held-out prompt compares one shared basis against phase-specific
bases; unlike a scalar gate, this comparison is invariant to residual scale.
"""

import argparse
import json
import math
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed

from opensora.schedulers.iddpm import forward_with_cfg
from opensora.utils.misc import to_torch_dtype
from qdiff.models.quant_layer import QuantLayer
from tools.profile_stage_numeric_mechanism import (
    build_quant_model,
    prepare_conditioning,
    set_quant_mode,
)


OPERATORS = {
    "spatial_attention": "attn",
    "temporal_attention": "attn_temp",
    "cross_attention": "cross_attn",
    "ffn": "mlp",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--calib-config", required=True)
    parser.add_argument("--quant-ckpt", required=True)
    parser.add_argument("--prompt-path", required=True)
    parser.add_argument("--text-embeds", required=True)
    parser.add_argument("--prompt-indices", default="0,2,6")
    parser.add_argument("--holdout-prompt-index", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selected-progress", default="5,15,25,45,65,85,95")
    parser.add_argument(
        "--phase-groups",
        default="early:5,15,25;middle:45,65;late:85,95",
        help="Semicolon-separated phase:name lists; every selected progress must occur exactly once.",
    )
    parser.add_argument("--selected-blocks", default="0,9,17,27")
    parser.add_argument("--operators", default=",".join(OPERATORS))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--svd-oversample", type=int, default=8)
    parser.add_argument("--svd-niter", type=int, default=3)
    parser.add_argument("--time-mp-config-weight", required=True)
    parser.add_argument("--time-mp-config-act", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_int_set(value):
    return sorted({int(x) for x in value.split(",") if x.strip()})


def parse_phases(value, selected):
    phases = {}
    progress_to_phase = {}
    for item in value.split(";"):
        name, raw = item.split(":", 1)
        name = name.strip()
        points = parse_int_set(raw)
        if not name or not points:
            raise ValueError(f"Invalid phase group: {item!r}")
        phases[name] = points
        for progress in points:
            if progress in progress_to_phase:
                raise ValueError(f"Progress {progress} occurs in multiple phases")
            progress_to_phase[progress] = name
    if set(progress_to_phase) != set(selected):
        raise ValueError(
            f"Phase groups cover {sorted(progress_to_phase)}, selected progress is {selected}"
        )
    return phases, progress_to_phase


def assert_full_precision_state(qnn):
    bad = [
        name
        for name, module in qnn.named_modules()
        if isinstance(module, QuantLayer)
        and (getattr(module, "weight_quant", False) or getattr(module, "act_quant", False))
    ]
    if bad:
        raise RuntimeError(f"Trajectory state leak: quantization remains enabled for {bad[:8]}")


def first_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            result = first_tensor(item)
            if result is not None:
                return result
    if isinstance(output, dict):
        for item in output.values():
            result = first_tensor(item)
            if result is not None:
                return result
    return None


def evenly_spaced_indices(length, count, device):
    count = min(length, count)
    if count == length:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, count, device=device).round().long().unique()


class OutputCollector:
    def __init__(self, qnn, blocks, operators, max_tokens):
        self.branch = None
        self.active = False
        self.max_tokens = max_tokens
        self.captured = {}
        self.handles = []
        module_map = dict(qnn.named_modules())
        for block in blocks:
            for operator in operators:
                name = f"model.blocks.{block}.{OPERATORS[operator]}"
                if name not in module_map:
                    raise KeyError(f"Selected operator module not found: {name}")
                self.handles.append(
                    module_map[name].register_forward_hook(self._make_hook(block, operator, name))
                )

    def _make_hook(self, block, operator, name):
        def hook(module, inputs, output):
            if not self.active:
                return
            tensor = first_tensor(output)
            if tensor is None or tensor.ndim < 2:
                raise RuntimeError(f"No token tensor found at {name}")
            rows = tensor.detach().reshape(-1, tensor.shape[-1])
            indices = evenly_spaced_indices(rows.shape[0], self.max_tokens, rows.device)
            self.captured[(self.branch, block, operator)] = (
                rows.index_select(0, indices).float().cpu()
            )
        return hook

    def reset(self):
        self.captured = {}

    def close(self):
        for handle in self.handles:
            handle.remove()


def split_forward(qnn, x, timestep, conditioning, collector):
    y = conditioning["y"]
    mask = conditioning["mask"]
    y_shape = y.shape
    y = y.reshape([2, y_shape[0] // 2] + list(y_shape[1:]))
    timestep = timestep.reshape([2, -1])
    y_cond, y_uncond = y.unbind(0)
    t_cond, t_uncond = timestep.unbind(0)
    half = x[: len(x) // 2]
    collector.reset()
    collector.active = True
    for branch, branch_y, branch_t in (
        ("conditional", y_cond, t_cond),
        ("unconditional", y_uncond, t_uncond),
    ):
        collector.branch = branch
        qnn.forward(half, branch_t, branch_y, mask=mask)
    collector.active = False
    return dict(collector.captured)


def residual_metrics(residual, fp_output):
    residual_energy = float(residual.square().sum())
    fp_energy = float(fp_output.square().sum())
    return {
        "sampled_tokens": int(residual.shape[0]),
        "channels": int(residual.shape[1]),
        "residual_energy": residual_energy,
        "fp_output_energy": fp_energy,
        "relative_l2": math.sqrt(residual_energy / max(fp_energy, 1e-20)),
        "residual_rms": float(residual.square().mean().sqrt()),
    }


def fit_basis(matrix, rank, oversample, niter, device):
    matrix = matrix.to(device=device, dtype=torch.float32)
    max_q = min(matrix.shape)
    q = min(max_q, rank + oversample)
    if q < 1:
        raise ValueError(f"Cannot fit basis to matrix with shape {tuple(matrix.shape)}")
    _, singular, vectors = torch.pca_lowrank(matrix, q=q, center=False, niter=niter)
    use_rank = min(rank, vectors.shape[1])
    basis = vectors[:, :use_rank].contiguous()
    total_energy = matrix.square().sum().clamp_min(1e-20)
    captured = singular.square().cumsum(0) / total_energy
    return basis.cpu(), singular.cpu(), captured.cpu(), float(total_energy)


def reconstruction_energy(matrix, basis, device):
    matrix = matrix.to(device=device, dtype=torch.float32)
    basis = basis.to(device=device, dtype=torch.float32)
    projected = torch.matmul(matrix, basis)
    residual = matrix - torch.matmul(projected, basis.T)
    return float(residual.square().sum()), float(matrix.square().sum())


def subspace_similarity(left, right):
    cross = torch.matmul(left.T.float(), right.float())
    cosines = torch.linalg.svdvals(cross).clamp(0, 1)
    return {
        "mean_squared_principal_cosine": float(cosines.square().mean()),
        "mean_principal_cosine": float(cosines.mean()),
        "minimum_principal_cosine": float(cosines.min()),
        "maximum_principal_cosine": float(cosines.max()),
        "principal_cosines": [float(x) for x in cosines],
    }


def write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    selected = parse_int_set(args.selected_progress)
    blocks = parse_int_set(args.selected_blocks)
    prompt_indices = parse_int_set(args.prompt_indices)
    operators = [x.strip() for x in args.operators.split(",") if x.strip()]
    unknown = sorted(set(operators) - set(OPERATORS))
    if unknown:
        raise ValueError(f"Unknown operators: {unknown}; choose from {sorted(OPERATORS)}")
    phases, progress_to_phase = parse_phases(args.phase_groups, selected)
    if args.holdout_prompt_index not in prompt_indices:
        raise ValueError("The holdout prompt must be included in --prompt-indices")
    train_prompts = [x for x in prompt_indices if x != args.holdout_prompt_index]
    if not train_prompts:
        raise ValueError("At least one non-holdout prompt is required")

    cfg = Config.fromfile(args.config)
    cfg.multi_resolution = cfg.get("multi_resolution", False)
    dtype = to_torch_dtype(cfg.dtype)
    device = torch.device("cuda")
    set_random_seed(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    scheduler, latent_size, qnn = build_quant_model(args, cfg, device, dtype)
    collector = OutputCollector(qnn, blocks, operators, args.max_tokens)
    residual_records = []
    residual_rows = []
    total_steps = scheduler.num_timesteps

    for prompt_index in prompt_indices:
        conditioning = prepare_conditioning(args.text_embeds, prompt_index, device, dtype)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        init_noise = torch.randn(
            1, qnn.in_channels, *latent_size, device=device, generator=generator
        )
        z = torch.cat([init_noise, init_noise], dim=0)
        set_quant_mode(qnn, False, False)
        fp_model = partial(
            forward_with_cfg, qnn, cfg_scale=cfg.scheduler.cfg_scale, return_trajectory=False
        )
        trajectory = scheduler.ddim_sample_loop_progressive(
            fp_model,
            z.shape,
            noise=z,
            clip_denoised=False,
            model_kwargs=conditioning,
            progress=True,
            device=device,
        )
        current_x = z
        for sampling_index, result in enumerate(trajectory):
            progress = sampling_index + 1
            x_t = current_x.detach()
            current_x = result["sample"]
            if progress not in selected:
                continue
            internal_timestep = total_steps - progress
            original_timestep = int(scheduler.timestep_map[internal_timestep])
            timestep = torch.tensor(
                [original_timestep] * x_t.shape[0], device=device, dtype=torch.long
            )

            set_quant_mode(qnn, False, False)
            fp_outputs = split_forward(qnn, x_t, timestep, conditioning, collector)
            set_quant_mode(qnn, True, True)
            quant_outputs = split_forward(qnn, x_t, timestep, conditioning, collector)
            set_quant_mode(qnn, False, False)
            assert_full_precision_state(qnn)
            if set(fp_outputs) != set(quant_outputs):
                raise RuntimeError("FP and W4A6 collector keys differ")

            for branch, block, operator in sorted(fp_outputs):
                fp_output = fp_outputs[(branch, block, operator)]
                quant_output = quant_outputs[(branch, block, operator)]
                if fp_output.shape != quant_output.shape:
                    raise RuntimeError(
                        f"Shape mismatch at {branch}/block{block}/{operator}: "
                        f"{tuple(fp_output.shape)} vs {tuple(quant_output.shape)}"
                    )
                residual = fp_output - quant_output
                record = {
                    "prompt_index": prompt_index,
                    "progress": progress,
                    "phase": progress_to_phase[progress],
                    "original_timestep": original_timestep,
                    "branch": branch,
                    "block": block,
                    "operator": operator,
                    **residual_metrics(residual, fp_output),
                }
                residual_rows.append(record)
                residual_records.append((record, residual))

    collector.close()

    grouped_train = defaultdict(list)
    grouped_holdout = defaultdict(list)
    for record, residual in residual_records:
        group = (record["branch"], record["block"], record["operator"])
        if record["prompt_index"] == args.holdout_prompt_index:
            grouped_holdout[group].append((record, residual))
        else:
            grouped_train[(group, record["phase"])].append(residual)
            grouped_train[(group, "__shared__")].append(residual)

    basis_rows = []
    bases = {}
    for (group, phase), matrices in sorted(grouped_train.items()):
        matrix = torch.cat(matrices, dim=0)
        basis, singular, captured, total_energy = fit_basis(
            matrix, args.rank, args.svd_oversample, args.svd_niter, device
        )
        bases[(group, phase)] = basis
        basis_rows.append(
            {
                "branch": group[0],
                "block": group[1],
                "operator": group[2],
                "basis_scope": "shared" if phase == "__shared__" else "phase",
                "phase": None if phase == "__shared__" else phase,
                "train_rows": int(matrix.shape[0]),
                "channels": int(matrix.shape[1]),
                "rank": int(basis.shape[1]),
                "total_residual_energy": total_energy,
                "singular_values": [float(x) for x in singular],
                "cumulative_energy_fraction": [float(x) for x in captured],
            }
        )

    similarity_rows = []
    phase_names = list(phases)
    groups = sorted(grouped_holdout)
    for group in groups:
        for i, left_phase in enumerate(phase_names):
            for right_phase in phase_names[i + 1 :]:
                left = bases[(group, left_phase)]
                right = bases[(group, right_phase)]
                similarity_rows.append(
                    {
                        "branch": group[0],
                        "block": group[1],
                        "operator": group[2],
                        "left_phase": left_phase,
                        "right_phase": right_phase,
                        **subspace_similarity(left, right),
                    }
                )

    comparison_rows = []
    for group, items in sorted(grouped_holdout.items()):
        shared_basis = bases[(group, "__shared__")]
        for record, matrix in items:
            phase_basis = bases[(group, record["phase"])]
            shared_error, energy = reconstruction_energy(matrix, shared_basis, device)
            phase_error, phase_energy = reconstruction_energy(matrix, phase_basis, device)
            if not math.isclose(energy, phase_energy, rel_tol=1e-5, abs_tol=1e-5):
                raise RuntimeError("Reconstruction energy mismatch")
            comparison_rows.append(
                {
                    **{k: record[k] for k in (
                        "prompt_index", "progress", "phase", "original_timestep",
                        "branch", "block", "operator",
                    )},
                    "residual_energy": energy,
                    "shared_reconstruction_error": shared_error,
                    "phase_reconstruction_error": phase_error,
                    "shared_unexplained_fraction": shared_error / max(energy, 1e-20),
                    "phase_unexplained_fraction": phase_error / max(energy, 1e-20),
                    "relative_error_reduction": (shared_error - phase_error)
                    / max(shared_error, 1e-20),
                }
            )

    shared_total = sum(row["shared_reconstruction_error"] for row in comparison_rows)
    phase_total = sum(row["phase_reconstruction_error"] for row in comparison_rows)
    positive_cells = sum(
        row["phase_reconstruction_error"] < row["shared_reconstruction_error"]
        for row in comparison_rows
    )
    overall_gain = (shared_total - phase_total) / max(shared_total, 1e-20)
    positive_fraction = positive_cells / max(len(comparison_rows), 1)
    median_overlap = float(
        torch.tensor(
            [row["mean_squared_principal_cosine"] for row in similarity_rows]
        ).median()
    )
    # Frozen before confirmatory execution: phase experts are worth pursuing
    # only if they improve held-out rank-r reconstruction by >=10%, improve at
    # least 75% of cells, and are not merely near-identical bases (>0.90 overlap).
    go = overall_gain >= 0.10 and positive_fraction >= 0.75 and median_overlap <= 0.90
    decision = {
        "decision": "go" if go else "kill_or_revise",
        "heldout_prompt_index": args.holdout_prompt_index,
        "rank": args.rank,
        "shared_total_reconstruction_error": shared_total,
        "phase_total_reconstruction_error": phase_total,
        "heldout_relative_error_reduction": overall_gain,
        "positive_cell_fraction": positive_fraction,
        "median_pairwise_subspace_overlap": median_overlap,
        "thresholds": {
            "minimum_relative_error_reduction": 0.10,
            "minimum_positive_cell_fraction": 0.75,
            "maximum_median_pairwise_subspace_overlap": 0.90,
        },
        "interpretation": (
            "Shared-basis reconstruction already absorbs pure residual-magnitude changes; "
            "a phase-specific gain therefore tests direction/subspace change rather than scalar strength."
        ),
    }

    write_jsonl(outdir / "residual_metrics.jsonl", residual_rows)
    write_jsonl(outdir / "basis_summaries.jsonl", basis_rows)
    write_jsonl(outdir / "subspace_similarity.jsonl", similarity_rows)
    write_jsonl(outdir / "heldout_basis_comparison.jsonl", comparison_rows)
    (outdir / "decision.json").write_text(json.dumps(decision, indent=2))
    metadata = {
        "prompt_indices": prompt_indices,
        "train_prompt_indices": train_prompts,
        "holdout_prompt_index": args.holdout_prompt_index,
        "seed": args.seed,
        "selected_progress": selected,
        "phases": phases,
        "selected_blocks": blocks,
        "operators": operators,
        "branches": ["conditional", "unconditional"],
        "max_tokens": args.max_tokens,
        "rank": args.rank,
        "runtime_dtype": str(dtype),
        "quant_checkpoint": args.quant_ckpt,
        "paired_input_design": True,
        "fp_state_restored_after_each_selected_progress": True,
        "row_counts": {
            "residual_metrics": len(residual_rows),
            "basis_summaries": len(basis_rows),
            "subspace_similarity": len(similarity_rows),
            "heldout_basis_comparison": len(comparison_rows),
        },
        "all_values_finite": all(
            math.isfinite(value)
            for rows in (residual_rows, basis_rows, similarity_rows, comparison_rows)
            for row in rows
            for value in row.values()
            if isinstance(value, float)
        ),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"metadata": metadata, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
