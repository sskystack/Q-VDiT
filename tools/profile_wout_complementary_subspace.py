#!/usr/bin/env python3
"""Profile whether temporal Wout needs a complementary output direction.

The released temporal TQE path uses a rank-1 Wout.  This paired-input,
no-forward-modification diagnostic fits additive residual corrections on
attn_temp.proj and compares equal-parameter rank-1 models:

  unconstrained:  Q(X) a_u b_u^T
  parallel:       Q(X) a_p b_1^T
  orthogonal:     Q(X) a_o b_o^T,  b_o^T b_1 = 0

where b_1 is the released Wout output direction.  It also compares the
structured rank-2 sum (parallel + orthogonal) with an unconstrained rank-2
control.  Both an unmasked additive branch and the released (1 + M) temporal
shape are tested.  Prompts 0 and 2 fit the models; prompt 6 is held out.
"""

import argparse
import gc
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
from tools.profile_stage_numeric_mechanism import (
    build_quant_model,
    prepare_conditioning,
    set_quant_mode,
)
from tools.profile_temporal_frequency_tqe import (
    TemporalFrequencyCollector,
    apply_model,
    assert_full_precision_state,
    block_index,
    energy,
    error_triplet,
    fit_reduced_rank_ridge,
    parse_int_set,
    parse_phases,
    project_ac,
    project_dc,
    split_forward,
    summarize_leakage,
    summarize_path_rows,
    tensor_dot,
    write_jsonl,
)


MODEL_NAMES = (
    "unconstrained_rank1",
    "parallel_rank1",
    "orthogonal_rank1",
    "shared_rank2",
    "parallel_plus_orthogonal_rank1x2",
)
BRANCH_SHAPES = ("unmasked", "released_mask_shape")


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
    )
    parser.add_argument("--selected-blocks", default="0,9,17,27")
    parser.add_argument("--max-trajectories", type=int, default=16)
    parser.add_argument("--ridge-ratio", type=float, default=1e-4)
    parser.add_argument("--pca-oversample", type=int, default=8)
    parser.add_argument("--pca-niter", type=int, default=6)
    parser.add_argument("--time-mp-config-weight", required=True)
    parser.add_argument("--time-mp-config-act", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def unit_vector(tensor, name):
    tensor = tensor.detach().float().reshape(-1)
    norm = torch.linalg.vector_norm(tensor)
    if not torch.isfinite(norm) or float(norm) <= 1e-12:
        raise RuntimeError(f"Degenerate vector for {name}: norm={float(norm)}")
    return tensor / norm, float(norm)


def extract_wout_geometry(collector):
    geometries = {}
    for name, module in sorted(collector.selected_modules.items()):
        if module.loraA_out.weight.shape[0] != 1 or module.loraB_out.weight.shape[1] != 1:
            raise RuntimeError(
                f"Expected released Wout rank 1 at {name}, got "
                f"A={tuple(module.loraA_out.weight.shape)}, "
                f"B={tuple(module.loraB_out.weight.shape)}"
            )
        input_unit, input_norm = unit_vector(
            module.loraA_out.weight[0], f"{name}.loraA_out"
        )
        output_unit, output_norm = unit_vector(
            module.loraB_out.weight[:, 0], f"{name}.loraB_out"
        )
        mask = module.mask.detach().float().cpu()
        geometries[name] = {
            "module": name,
            "block": block_index(name),
            "input_unit": input_unit.cpu(),
            "output_unit": output_unit.cpu(),
            "input_norm": input_norm,
            "output_norm": output_norm,
            "wout_frobenius_norm": input_norm * output_norm,
            "mask": mask,
            "mask_mean": float(mask.mean()),
            "mask_std": float(mask.std(unbiased=False)),
            "mask_min": float(mask.min()),
            "mask_max": float(mask.max()),
        }
    return geometries


def branch_gain(geometry, branch_shape, device, dtype):
    if branch_shape == "unmasked":
        return None
    if branch_shape == "released_mask_shape":
        return (1.0 + geometry["mask"]).to(device=device, dtype=dtype)
    raise ValueError(branch_shape)


def stack_samples(samples, reference, geometry, branch_shape):
    x3 = torch.cat([sample["x"] for sample in samples], dim=0).float()
    y3 = torch.cat([sample[reference] for sample in samples], dim=0).float()
    gain = branch_gain(geometry, branch_shape, x3.device, x3.dtype)
    if gain is not None:
        x3 = x3 * gain
    return x3.reshape(-1, x3.shape[-1]), y3.reshape(-1, y3.shape[-1])


def output_projection(y, output_unit):
    scores = torch.matmul(y, output_unit)
    parallel = scores.unsqueeze(-1) * output_unit.unsqueeze(0)
    return parallel, y - parallel


def force_parallel_model(model, output_unit):
    input_factors, output_basis = model
    scale = torch.dot(output_basis[:, 0], output_unit)
    if abs(float(scale)) <= 1e-8:
        raise RuntimeError("Parallel fit lost the fixed Wout output direction")
    return input_factors * scale, output_unit.unsqueeze(1).contiguous()


def force_orthogonal_model(model, output_unit):
    input_factors, output_basis = model
    projected = output_basis[:, 0] - output_unit * torch.dot(
        output_basis[:, 0], output_unit
    )
    norm = torch.linalg.vector_norm(projected)
    if float(norm) <= 1e-8:
        raise RuntimeError("Orthogonal fit collapsed onto the released Wout direction")
    return input_factors * norm, (projected / norm).unsqueeze(1).contiguous()


def vector_cosine(left, right):
    left = left.reshape(-1).double()
    right = right.reshape(-1).double()
    denominator = (
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    ).clamp_min(1e-30)
    return float(torch.dot(left, right) / denominator)


def model_alignment(model, geometry):
    input_factors, output_basis = model
    input_cosine = vector_cosine(
        input_factors[:, 0].cpu(), geometry["input_unit"]
    )
    output_cosine = vector_cosine(
        output_basis[:, 0].cpu(), geometry["output_unit"]
    )
    return {
        "input_cosine_with_released_wout": input_cosine,
        "output_cosine_with_released_wout": output_cosine,
        "absolute_output_cosine_with_released_wout": abs(output_cosine),
        "coefficient_cosine_with_released_wout": input_cosine * output_cosine,
    }


def pair_model_alignment(left, right):
    left_input, left_output = left
    right_input, right_output = right
    input_cosine = vector_cosine(left_input[:, 0], right_input[:, 0])
    output_cosine = vector_cosine(left_output[:, 0], right_output[:, 0])
    return {
        "input_absolute_cosine": abs(input_cosine),
        "output_absolute_cosine": abs(output_cosine),
        "coefficient_cosine": input_cosine * output_cosine,
    }


def gain_retention(candidate_gain, control_gain):
    if control_gain <= 0:
        return None
    return candidate_gain / control_gain


def fit_one_model(x, target, rank, seed, args, device):
    torch.manual_seed(seed)
    model, diagnostics = fit_reduced_rank_ridge(
        x, target, rank, args, device
    )
    return model, diagnostics


def fit_model_set(samples, reference, geometry, branch_shape, seed, args, device):
    x, y = stack_samples(samples, reference, geometry, branch_shape)
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)
    output_unit = geometry["output_unit"].to(device=device, dtype=torch.float32)
    y_parallel, y_orthogonal = output_projection(y, output_unit)
    models = {}
    fit_rows = []
    specifications = (
        ("unconstrained_rank1", y, 1),
        ("parallel_rank1", y_parallel, 1),
        ("orthogonal_rank1", y_orthogonal, 1),
        ("shared_rank2", y, 2),
    )
    for fit_index, (model_name, fit_target, rank) in enumerate(specifications):
        model, diagnostics = fit_one_model(
            x, fit_target, rank, seed + fit_index, args, device
        )
        if model_name == "parallel_rank1":
            model = force_parallel_model(model, output_unit)
        elif model_name == "orthogonal_rank1":
            model = force_orthogonal_model(model, output_unit)
        prediction = apply_model(x, model)
        full_target_error = energy(y - prediction)
        row = {
            "fit": model_name,
            "branch_shape": branch_shape,
            "full_target_train_error_energy": full_target_error,
            "full_target_train_error_reduction": (
                energy(y) - full_target_error
            ) / max(energy(y), 1e-30),
            **diagnostics,
            **model_alignment(model, geometry),
        }
        models[model_name] = model
        fit_rows.append(row)
    return models, fit_rows


def prediction_set(x3, geometry, branch_shape, models):
    gain = branch_gain(geometry, branch_shape, x3.device, x3.dtype)
    effective_x = x3 if gain is None else x3 * gain
    flat_x = effective_x.reshape(-1, effective_x.shape[-1])
    predictions = {
        name: apply_model(flat_x, models[name]).reshape(
            effective_x.shape[0], effective_x.shape[1], -1
        )
        for name in (
            "unconstrained_rank1",
            "parallel_rank1",
            "orthogonal_rank1",
            "shared_rank2",
        )
    }
    predictions["parallel_plus_orthogonal_rank1x2"] = (
        predictions["parallel_rank1"] + predictions["orthogonal_rank1"]
    )
    return predictions


def add_error_triplet(accumulator, prefix, residual):
    for component, value in error_triplet(residual).items():
        accumulator[f"{prefix}_{component}_error"] += value


def evaluate_models(module_name, samples, reference, geometry, branch_shape, models, device):
    scopes = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    output_unit = geometry["output_unit"].to(device=device, dtype=torch.float32)
    for sample in samples:
        x3 = sample["x"].to(device=device, dtype=torch.float32)
        target = sample[reference].to(device=device, dtype=torch.float32)
        predictions = prediction_set(x3, geometry, branch_shape, models)
        target_flat = target.reshape(-1, target.shape[-1])
        target_parallel, target_orthogonal = output_projection(
            target_flat, output_unit
        )
        scope_keys = {
            ("overall", "all"),
            ("phase", sample["phase"]),
            ("branch", sample["branch"]),
            ("phase_branch", f'{sample["phase"]}:{sample["branch"]}'),
        }
        for scope_key in scope_keys:
            values = scopes[scope_key]
            add_error_triplet(values, "released", target)
            for model_name, prediction in predictions.items():
                add_error_triplet(values, model_name, target - prediction)
            values["target_wout_parallel_energy"] += energy(target_parallel)
            values["target_wout_orthogonal_energy"] += energy(target_orthogonal)
            counts[scope_key] += 1

    rows = []
    for (scope, group), values in sorted(scopes.items()):
        row = {
            "module": module_name,
            "block": block_index(module_name),
            "reference": reference,
            "branch_shape": branch_shape,
            "scope": scope,
            "group": group,
            "heldout_cells": counts[(scope, group)],
            **dict(values),
        }
        for model_name in MODEL_NAMES:
            for component in ("total", "dc", "ac"):
                released = row[f"released_{component}_error"]
                candidate = row[f"{model_name}_{component}_error"]
                row[f"{model_name}_{component}_gain_vs_released"] = (
                    released - candidate
                ) / max(released, 1e-30)
        for component in ("total", "dc", "ac"):
            shared = row[f"shared_rank2_{component}_error"]
            structured = row[
                f"parallel_plus_orthogonal_rank1x2_{component}_error"
            ]
            row[f"structured_rank2_{component}_gain_vs_shared_rank2"] = (
                shared - structured
            ) / max(shared, 1e-30)
        unconstrained_gain = row[
            "unconstrained_rank1_total_gain_vs_released"
        ]
        orthogonal_gain = row["orthogonal_rank1_total_gain_vs_released"]
        row["orthogonal_rank1_gain_retention_vs_unconstrained_rank1"] = (
            gain_retention(orthogonal_gain, unconstrained_gain)
        )
        total_target_energy = (
            row["target_wout_parallel_energy"]
            + row["target_wout_orthogonal_energy"]
        )
        row["target_wout_parallel_energy_fraction"] = (
            row["target_wout_parallel_energy"]
            / max(total_target_energy, 1e-30)
        )
        rows.append(row)
    return rows


def prompt_stability(module_name, train_samples, reference, geometry, args, device):
    prompts = sorted({sample["prompt_index"] for sample in train_samples})
    if len(prompts) < 2:
        return []
    prompt_models = {}
    for prompt in prompts:
        selected = [
            sample for sample in train_samples if sample["prompt_index"] == prompt
        ]
        models, _ = fit_model_set(
            selected,
            reference,
            geometry,
            "unmasked",
            args.seed + 10000 + block_index(module_name) * 100 + prompt * 10,
            args,
            device,
        )
        prompt_models[prompt] = models
    rows = []
    for left_index, left_prompt in enumerate(prompts):
        for right_prompt in prompts[left_index + 1 :]:
            for model_name in ("unconstrained_rank1", "orthogonal_rank1"):
                rows.append(
                    {
                        "module": module_name,
                        "block": block_index(module_name),
                        "reference": reference,
                        "left_prompt": left_prompt,
                        "right_prompt": right_prompt,
                        "fit": model_name,
                        **pair_model_alignment(
                            prompt_models[left_prompt][model_name],
                            prompt_models[right_prompt][model_name],
                        ),
                    }
                )
    return rows


def run_fits(rank_samples, geometries, holdout_prompt, args, device):
    fit_rows = []
    evaluation_rows = []
    stability_rows = []
    for module_name, samples in sorted(rank_samples.items()):
        train = [
            sample for sample in samples
            if sample["prompt_index"] != holdout_prompt
        ]
        heldout = [
            sample for sample in samples
            if sample["prompt_index"] == holdout_prompt
        ]
        if not train or not heldout:
            raise RuntimeError(f"Missing train/heldout samples for {module_name}")
        geometry = geometries[module_name]
        for reference_index, reference in enumerate(
            ("local_same_input", "fp_trajectory")
        ):
            for shape_index, branch_shape in enumerate(BRANCH_SHAPES):
                seed = (
                    args.seed
                    + block_index(module_name) * 100
                    + reference_index * 20
                    + shape_index * 10
                )
                models, rows = fit_model_set(
                    train,
                    reference,
                    geometry,
                    branch_shape,
                    seed,
                    args,
                    device,
                )
                for row in rows:
                    fit_rows.append(
                        {
                            "module": module_name,
                            "block": block_index(module_name),
                            "reference": reference,
                            **row,
                        }
                    )
                evaluation_rows.extend(
                    evaluate_models(
                        module_name,
                        heldout,
                        reference,
                        geometry,
                        branch_shape,
                        models,
                        device,
                    )
                )
                del models
                torch.cuda.empty_cache()
            stability_rows.extend(
                prompt_stability(
                    module_name,
                    train,
                    reference,
                    geometry,
                    args,
                    device,
                )
            )
    return fit_rows, evaluation_rows, stability_rows


def aggregate_evaluation(rows, reference, branch_shape, scope, group):
    selected = [
        row for row in rows
        if row["reference"] == reference
        and row["branch_shape"] == branch_shape
        and row["scope"] == scope
        and row["group"] == group
    ]
    if not selected:
        raise RuntimeError(
            f"No evaluation rows for {reference}/{branch_shape}/{scope}/{group}"
        )
    output = {"modules": len(selected)}
    for model_name in ("released", *MODEL_NAMES):
        for component in ("total", "dc", "ac"):
            output[f"{model_name}_{component}_error"] = sum(
                row[f"{model_name}_{component}_error"] for row in selected
            )
    output["target_wout_parallel_energy"] = sum(
        row["target_wout_parallel_energy"] for row in selected
    )
    output["target_wout_orthogonal_energy"] = sum(
        row["target_wout_orthogonal_energy"] for row in selected
    )
    for model_name in MODEL_NAMES:
        for component in ("total", "dc", "ac"):
            released = output[f"released_{component}_error"]
            candidate = output[f"{model_name}_{component}_error"]
            output[f"{model_name}_{component}_gain_vs_released"] = (
                released - candidate
            ) / max(released, 1e-30)
    for component in ("total", "dc", "ac"):
        shared = output[f"shared_rank2_{component}_error"]
        structured = output[
            f"parallel_plus_orthogonal_rank1x2_{component}_error"
        ]
        output[f"structured_rank2_{component}_gain_vs_shared_rank2"] = (
            shared - structured
        ) / max(shared, 1e-30)
    unconstrained_gain = output[
        "unconstrained_rank1_total_gain_vs_released"
    ]
    orthogonal_gain = output["orthogonal_rank1_total_gain_vs_released"]
    output["orthogonal_rank1_gain_retention_vs_unconstrained_rank1"] = (
        gain_retention(orthogonal_gain, unconstrained_gain)
    )
    total_target = (
        output["target_wout_parallel_energy"]
        + output["target_wout_orthogonal_energy"]
    )
    output["target_wout_parallel_energy_fraction"] = (
        output["target_wout_parallel_energy"] / max(total_target, 1e-30)
    )
    return output


def make_decision(evaluation_rows):
    primary = aggregate_evaluation(
        evaluation_rows,
        "local_same_input",
        "unmasked",
        "overall",
        "all",
    )
    trajectory = aggregate_evaluation(
        evaluation_rows,
        "fp_trajectory",
        "unmasked",
        "overall",
        "all",
    )
    mask_shaped = aggregate_evaluation(
        evaluation_rows,
        "local_same_input",
        "released_mask_shape",
        "overall",
        "all",
    )
    local_modules = [
        row for row in evaluation_rows
        if row["reference"] == "local_same_input"
        and row["branch_shape"] == "unmasked"
        and row["scope"] == "overall"
        and row["group"] == "all"
    ]
    positive_blocks = sorted(
        row["block"]
        for row in local_modules
        if row["orthogonal_rank1_total_gain_vs_released"] > 0
        and row["orthogonal_rank1_dc_gain_vs_released"] >= -0.01
        and row["orthogonal_rank1_ac_gain_vs_released"] >= -0.01
    )
    branch_values = {}
    for branch in ("conditional", "unconditional"):
        branch_values[branch] = aggregate_evaluation(
            evaluation_rows,
            "fp_trajectory",
            "unmasked",
            "branch",
            branch,
        )
    shared_gain = primary["shared_rank2_total_gain_vs_released"]
    structured_gain = primary[
        "parallel_plus_orthogonal_rank1x2_total_gain_vs_released"
    ]
    structured_retention = gain_retention(structured_gain, shared_gain)
    orthogonal_retention = primary[
        "orthogonal_rank1_gain_retention_vs_unconstrained_rank1"
    ]
    thresholds = {
        "minimum_unconstrained_rank1_total_gain_vs_released": 0.10,
        "minimum_orthogonal_rank1_total_gain_vs_released": 0.15,
        "minimum_orthogonal_gain_retention_vs_unconstrained_rank1": 0.80,
        "minimum_orthogonal_dc_gain_vs_released": -0.01,
        "minimum_orthogonal_ac_gain_vs_released": -0.01,
        "minimum_positive_blocks": 3,
        "minimum_structured_rank2_gain_retention_vs_shared_rank2": 0.90,
        "minimum_trajectory_orthogonal_total_gain_vs_released": -0.01,
        "minimum_each_trajectory_branch_gain_vs_released": -0.01,
    }
    go = (
        primary["unconstrained_rank1_total_gain_vs_released"]
        >= thresholds["minimum_unconstrained_rank1_total_gain_vs_released"]
        and primary["orthogonal_rank1_total_gain_vs_released"]
        >= thresholds["minimum_orthogonal_rank1_total_gain_vs_released"]
        and orthogonal_retention is not None
        and orthogonal_retention
        >= thresholds[
            "minimum_orthogonal_gain_retention_vs_unconstrained_rank1"
        ]
        and primary["orthogonal_rank1_dc_gain_vs_released"]
        >= thresholds["minimum_orthogonal_dc_gain_vs_released"]
        and primary["orthogonal_rank1_ac_gain_vs_released"]
        >= thresholds["minimum_orthogonal_ac_gain_vs_released"]
        and len(positive_blocks) >= thresholds["minimum_positive_blocks"]
        and structured_retention is not None
        and structured_retention
        >= thresholds[
            "minimum_structured_rank2_gain_retention_vs_shared_rank2"
        ]
        and trajectory["orthogonal_rank1_total_gain_vs_released"]
        >= thresholds["minimum_trajectory_orthogonal_total_gain_vs_released"]
        and all(
            values["orthogonal_rank1_total_gain_vs_released"]
            >= thresholds["minimum_each_trajectory_branch_gain_vs_released"]
            for values in branch_values.values()
        )
    )
    return {
        "decision": "go" if go else "kill_or_revise",
        "meaning": (
            "Promote a complementary Wout direction only if an output-orthogonal "
            "rank-1 correction retains most unconstrained rank-1 headroom, improves "
            "held-out local residuals without DC/AC regressions, remains trajectory-safe, "
            "and parallel+orthogonal rank-2 matches the unconstrained rank-2 control."
        ),
        "primary_unmasked_local_same_input": primary,
        "unmasked_fp_trajectory": trajectory,
        "released_mask_shape_local_same_input": mask_shaped,
        "positive_blocks": positive_blocks,
        "trajectory_branches": branch_values,
        "structured_rank2_gain_retention_vs_shared_rank2": structured_retention,
        "thresholds": thresholds,
    }


def geometry_rows(geometries):
    rows = []
    for geometry in geometries.values():
        rows.append(
            {
                key: value
                for key, value in geometry.items()
                if key not in ("input_unit", "output_unit", "mask")
            }
        )
    return rows


def all_float_values_finite(rows):
    return all(
        math.isfinite(value)
        for row in rows
        for value in row.values()
        if isinstance(value, float)
    )


def main():
    args = parse_args()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    selected = parse_int_set(args.selected_progress)
    blocks = parse_int_set(args.selected_blocks)
    prompt_indices = parse_int_set(args.prompt_indices)
    phases, progress_to_phase = parse_phases(args.phase_groups, selected)
    if args.holdout_prompt_index not in prompt_indices:
        raise ValueError("Holdout prompt must be included in --prompt-indices")
    train_prompts = [
        prompt for prompt in prompt_indices
        if prompt != args.holdout_prompt_index
    ]
    if not train_prompts:
        raise ValueError("At least one training prompt is required")

    cfg = Config.fromfile(args.config)
    cfg.multi_resolution = cfg.get("multi_resolution", False)
    dtype = to_torch_dtype(cfg.dtype)
    device = torch.device("cuda")
    set_random_seed(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    scheduler, latent_size, qnn = build_quant_model(
        args, cfg, device, dtype
    )
    collector = TemporalFrequencyCollector(
        qnn, blocks, ["proj"], args.max_trajectories
    )
    total_steps = scheduler.num_timesteps
    for prompt_index in prompt_indices:
        conditioning = prepare_conditioning(
            args.text_embeds, prompt_index, device, dtype
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        init_noise = torch.randn(
            1,
            qnn.in_channels,
            *latent_size,
            device=device,
            generator=generator,
        )
        z = torch.cat([init_noise, init_noise], dim=0)
        set_quant_mode(qnn, False, False)
        fp_model = partial(
            forward_with_cfg,
            qnn,
            cfg_scale=cfg.scheduler.cfg_scale,
            return_trajectory=False,
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
            collector.reset_progress()
            internal_timestep = total_steps - progress
            original_timestep = int(
                scheduler.timestep_map[internal_timestep]
            )
            timestep = torch.tensor(
                [original_timestep] * x_t.shape[0],
                device=device,
                dtype=torch.long,
            )
            phase = progress_to_phase[progress]
            set_quant_mode(qnn, False, False)
            split_forward(
                qnn,
                x_t,
                timestep,
                conditioning,
                collector,
                "fp",
                prompt_index,
                progress,
                phase,
                original_timestep,
            )
            set_quant_mode(qnn, True, True)
            split_forward(
                qnn,
                x_t,
                timestep,
                conditioning,
                collector,
                "w4a6",
                prompt_index,
                progress,
                phase,
                original_timestep,
            )
            set_quant_mode(qnn, False, False)
            assert_full_precision_state(qnn)

    collector.close()
    path_rows = collector.path_rows
    leakage_rows = collector.leakage_rows
    rank_samples = collector.rank_samples
    geometries = extract_wout_geometry(collector)
    selected_module_count = len(collector.selected_modules)
    encountered_modules = sorted(
        collector.encountered_quantized_modules
    )
    collector.release_model_refs()
    del qnn, collector
    gc.collect()
    torch.cuda.empty_cache()

    fit_rows, evaluation_rows, stability_rows = run_fits(
        rank_samples,
        geometries,
        args.holdout_prompt_index,
        args,
        device,
    )
    path_summaries = summarize_path_rows(path_rows)
    leakage_summaries = summarize_leakage(leakage_rows)
    decision = make_decision(evaluation_rows)
    max_reconstruction_error = max(
        row["actual_forward_reconstruction_relative_l2"]
        for row in path_rows
    )
    finite = all(
        all_float_values_finite(rows)
        for rows in (
            path_rows,
            leakage_rows,
            fit_rows,
            evaluation_rows,
            stability_rows,
        )
    )
    if max_reconstruction_error > 5e-3:
        decision["decision"] = "invalid"
        decision["invalid_reason"] = (
            "Released forward reconstruction relative L2 exceeded tolerance: "
            f"{max_reconstruction_error:.6g}"
        )
    elif not finite:
        decision["decision"] = "invalid"
        decision["invalid_reason"] = "Non-finite profiling value detected"

    write_jsonl(outdir / "wout_geometry.jsonl", geometry_rows(geometries))
    write_jsonl(outdir / "frequency_path_metrics.jsonl", path_rows)
    write_jsonl(outdir / "frequency_path_summaries.jsonl", path_summaries)
    write_jsonl(outdir / "mask_frequency_leakage.jsonl", leakage_rows)
    write_jsonl(
        outdir / "mask_frequency_leakage_summaries.jsonl",
        leakage_summaries,
    )
    write_jsonl(outdir / "rank_fit_diagnostics.jsonl", fit_rows)
    write_jsonl(
        outdir / "heldout_complementary_subspace_comparison.jsonl",
        evaluation_rows,
    )
    write_jsonl(
        outdir / "train_prompt_subspace_stability.jsonl",
        stability_rows,
    )
    (outdir / "decision.json").write_text(
        json.dumps(decision, indent=2)
    )
    metadata = {
        "prompt_indices": prompt_indices,
        "train_prompt_indices": train_prompts,
        "holdout_prompt_index": args.holdout_prompt_index,
        "seed": args.seed,
        "selected_progress": selected,
        "phases": phases,
        "selected_blocks": blocks,
        "selected_role": "proj",
        "max_trajectories": args.max_trajectories,
        "runtime_dtype": str(dtype),
        "quant_checkpoint": args.quant_ckpt,
        "released_forward_unchanged": True,
        "paired_fp_trajectory": True,
        "released_wout_rank": 1,
        "rank1_equal_parameter_controls": [
            "unconstrained",
            "parallel_to_released_wout_output",
            "orthogonal_to_released_wout_output",
        ],
        "rank2_equal_parameter_controls": [
            "unconstrained_shared_rank2",
            "parallel_rank1_plus_orthogonal_rank1",
        ],
        "branch_shapes": {
            "unmasked": "C",
            "released_mask_shape": "C + M * C",
        },
        "fit_method": (
            "ridge-whitened globally optimal reduced-rank regression"
        ),
        "ridge_ratio": args.ridge_ratio,
        "pca_oversample": args.pca_oversample,
        "pca_niter": args.pca_niter,
        "selected_module_count": selected_module_count,
        "encountered_quantized_module_count": len(encountered_modules),
        "encountered_quantized_modules": encountered_modules,
        "maximum_actual_forward_reconstruction_relative_l2": (
            max_reconstruction_error
        ),
        "all_values_finite": finite,
        "row_counts": {
            "wout_geometry": len(geometries),
            "frequency_path_metrics": len(path_rows),
            "frequency_path_summaries": len(path_summaries),
            "mask_frequency_leakage": len(leakage_rows),
            "mask_frequency_leakage_summaries": len(leakage_summaries),
            "rank_fit_diagnostics": len(fit_rows),
            "heldout_complementary_subspace_comparison": len(
                evaluation_rows
            ),
            "train_prompt_subspace_stability": len(stability_rows),
        },
    }
    (outdir / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(json.dumps({"metadata": metadata, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
