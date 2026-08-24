#!/usr/bin/env python3
"""Profile temporal DC/AC structure in the released Q-VDiT TQE paths.

This is a paired-input, no-forward-modification diagnostic.  For temporal
attention q/k/v/proj linears it measures how Win, the released unmasked Wout,
and the released masked Wout affect the temporal DC and AC residuals.  The
primary module test is restricted to attn_temp.proj and compares equal-parameter
ridge-whitened reduced-rank fits:

  shared rank-2:       Q(X) W_shared^T
  frequency-decoupled: P_DC Q(X) W_DC^T + P_AC Q(X) W_AC^T, rank 1 each

Fits use train prompts only and are evaluated on a held-out prompt.  Both a
same-input local FP target and the actual paired FP-trajectory layer output are
reported, preventing upstream quantization drift from being mistaken for a
local TQE mechanism.
"""

import argparse
import gc
import json
import math
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import set_random_seed

from opensora.schedulers.iddpm import forward_with_cfg
from opensora.utils.misc import to_torch_dtype
from qdiff.models.quant_layer import QuantLayer
from qdiff.models.stdit_quant_layer import QuantTemporalAttnLinear
from qdiff.quantizer.base_quantizer import StraightThrough
from tools.profile_stage_numeric_mechanism import (
    build_quant_model,
    prepare_conditioning,
    set_quant_mode,
)


TEMPORAL_ROLES = ("q", "k", "v", "proj")


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
    parser.add_argument("--temporal-roles", default=",".join(TEMPORAL_ROLES))
    parser.add_argument("--max-trajectories", type=int, default=16)
    parser.add_argument("--ridge-ratio", type=float, default=1e-4)
    parser.add_argument("--pca-oversample", type=int, default=4)
    parser.add_argument("--pca-niter", type=int, default=4)
    parser.add_argument("--time-mp-config-weight", required=True)
    parser.add_argument("--time-mp-config-act", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_int_set(value):
    return sorted({int(item) for item in value.split(",") if item.strip()})


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
                raise ValueError(f"Progress {progress} belongs to multiple phases")
            progress_to_phase[progress] = name
    if set(progress_to_phase) != set(selected):
        raise ValueError(
            f"Phase groups cover {sorted(progress_to_phase)}, selected progress is {selected}"
        )
    return phases, progress_to_phase


def block_index(name):
    parts = name.split(".")
    if "blocks" not in parts:
        return None
    pos = parts.index("blocks")
    if pos + 1 < len(parts) and parts[pos + 1].isdigit():
        return int(parts[pos + 1])
    return None


def temporal_role(name):
    marker = ".attn_temp."
    if marker not in name:
        return None
    role = name.split(marker, 1)[1]
    return role if role in TEMPORAL_ROLES else None


def evenly_spaced_indices(length, count, device):
    count = min(length, count)
    if count == length:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, count, device=device).round().long().unique()


def project_dc(tensor):
    mean = tensor.mean(dim=1, keepdim=True)
    return mean.expand_as(tensor)


def project_ac(tensor):
    return tensor - project_dc(tensor)


def project_component(tensor, component):
    if component == "dc":
        return project_dc(tensor)
    if component == "ac":
        return project_ac(tensor)
    raise ValueError(component)


def energy(tensor):
    return float(torch.sum(tensor.double().square()))


def tensor_dot(left, right):
    return float(torch.sum(left.double() * right.double()))


def cosine(path, target):
    path_energy = energy(path)
    target_energy = energy(target)
    return tensor_dot(path, target) / math.sqrt(max(path_energy * target_energy, 1e-30))


def relative_l2(left, right):
    return float(
        torch.linalg.vector_norm((left - right).double())
        / torch.linalg.vector_norm(right.double()).clamp_min(1e-20)
    )


def write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def assert_full_precision_state(qnn):
    bad = [
        name
        for name, module in qnn.named_modules()
        if isinstance(module, QuantLayer)
        and (getattr(module, "weight_quant", False) or getattr(module, "act_quant", False))
    ]
    if bad:
        raise RuntimeError(f"Trajectory state leak: quantization remains enabled for {bad[:8]}")


def component_record(reference, component, target, raw, base, unmasked, masked, released):
    target_raw = project_component(target - raw, component)
    target_base = project_component(target - base, component)
    win_path = project_component(base - raw, component)
    unmasked_path = project_component(unmasked, component)
    masked_path = project_component(masked, component)
    total_path = unmasked_path + masked_path
    raw_error = energy(target_raw)
    base_error = energy(target_base)
    unmasked_error = energy(project_component(target - (base + unmasked), component))
    masked_error = energy(project_component(target - (base + masked), component))
    release_error = energy(project_component(target - released, component))
    return {
        "reference": reference,
        "component": component,
        "raw_error_energy": raw_error,
        "base_after_win_error_energy": base_error,
        "unmasked_only_error_energy": unmasked_error,
        "masked_only_error_energy": masked_error,
        "released_error_energy": release_error,
        "win_error_removal_ratio": (raw_error - base_error) / max(raw_error, 1e-30),
        "unmasked_error_removal_ratio": (base_error - unmasked_error) / max(base_error, 1e-30),
        "masked_error_removal_ratio": (base_error - masked_error) / max(base_error, 1e-30),
        "released_error_removal_ratio": (base_error - release_error) / max(base_error, 1e-30),
        "win_cosine": cosine(win_path, target_raw),
        "unmasked_cosine": cosine(unmasked_path, target_base),
        "masked_cosine": cosine(masked_path, target_base),
        "total_wout_cosine": cosine(total_path, target_base),
        "win_overcompensates": base_error > raw_error,
        "unmasked_overcompensates": unmasked_error > base_error,
        "masked_overcompensates": masked_error > base_error,
        "released_overcompensates": release_error > base_error,
    }


class TemporalFrequencyCollector:
    def __init__(self, qnn, blocks, roles, max_trajectories):
        self.qnn = qnn
        self.blocks = set(blocks)
        self.roles = set(roles)
        self.max_trajectories = max_trajectories
        self.active = False
        self.mode = None
        self.branch = None
        self.prompt_index = None
        self.progress = None
        self.phase = None
        self.original_timestep = None
        self.call_counts = defaultdict(int)
        self.current_call = {}
        self.raw_inputs = {}
        self.quant_inputs = {}
        self.quant_weights = {}
        self.inside_weight_recompute = False
        self.fp_samples = {}
        self.weight_cache = {}
        self.path_rows = []
        self.leakage_rows = []
        self.rank_samples = defaultdict(list)
        self.selected_modules = {}
        self.encountered_quantized_modules = set()
        self.handles = []
        self._register()

    def _register(self):
        for name, module in self.qnn.named_modules():
            if not isinstance(module, QuantTemporalAttnLinear):
                continue
            block = block_index(name)
            role = temporal_role(name)
            if block not in self.blocks or role not in self.roles:
                continue
            if module.weight.ndim != 2 or module.fwd_func is not F.linear:
                raise RuntimeError(f"Unsupported temporal QuantLayer: {name}")
            if not isinstance(module.activation_function, StraightThrough):
                raise RuntimeError(f"Temporal additive decomposition requires identity activation: {name}")
            if module.split != 0 or module.smooth_quant:
                raise RuntimeError(f"Split/SmoothQuant temporal layer is unsupported: {name}")
            self.selected_modules[name] = module
            self.handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
            self.handles.append(module.register_forward_hook(self._module_hook(name)))
            self.handles.append(module.act_quantizer.register_forward_hook(self._act_hook(name)))
            self.handles.append(module.weight_quantizer.register_forward_hook(self._weight_hook(name)))
        if not self.selected_modules:
            raise RuntimeError("No temporal q/k/v/proj QuantLayers matched the selection")

    def reset_progress(self):
        self.fp_samples.clear()

    def begin_pass(self, mode, branch, prompt_index, progress, phase, original_timestep):
        self.mode = mode
        self.branch = branch
        self.prompt_index = prompt_index
        self.progress = progress
        self.phase = phase
        self.original_timestep = original_timestep
        self.call_counts.clear()
        self.current_call.clear()
        self.raw_inputs.clear()
        self.quant_inputs.clear()
        self.quant_weights.clear()
        self.active = True

    def end_pass(self):
        self.active = False
        self.current_call.clear()
        self.raw_inputs.clear()
        self.quant_inputs.clear()
        self.quant_weights.clear()

    def _pre_hook(self, name):
        def hook(module, inputs):
            if not self.active or not inputs or not torch.is_tensor(inputs[0]):
                return
            call_index = self.call_counts[name]
            self.call_counts[name] += 1
            self.current_call[name] = call_index
            if self.mode == "w4a6":
                self.raw_inputs[name] = inputs[0].detach()
        return hook

    def _act_hook(self, name):
        def hook(module, inputs, output):
            if self.active and self.mode == "w4a6" and torch.is_tensor(output):
                self.quant_inputs[name] = output.detach()
        return hook

    def _weight_hook(self, name):
        def hook(module, inputs, output):
            if (
                self.active
                and self.mode == "w4a6"
                and not self.inside_weight_recompute
                and torch.is_tensor(output)
            ):
                self.quant_weights[name] = output.detach()
        return hook

    def _counterfactual_weights(self, name, module):
        n_bits = int(module.weight_quantizer.n_bits)
        bit_idx = int(getattr(module.weight_quantizer, "bit_idx", 0))
        key = (name, n_bits, bit_idx)
        if key not in self.weight_cache:
            wout = torch.matmul(module.loraB_out.weight, module.loraA_out.weight)
            self.inside_weight_recompute = True
            try:
                qweight_raw = module.weight_quantizer(module.weight)
            finally:
                self.inside_weight_recompute = False
            self.weight_cache[key] = (qweight_raw.detach(), wout.detach())
        return self.weight_cache[key]

    def _module_hook(self, name):
        def hook(module, inputs, output):
            if not self.active or not torch.is_tensor(output):
                return
            call_index = self.current_call.pop(name)
            key = (self.branch, name, call_index)
            if output.ndim != 3 or output.shape[1] != module.mask.shape[1]:
                raise RuntimeError(f"Unexpected temporal output shape at {name}: {tuple(output.shape)}")
            indices = evenly_spaced_indices(output.shape[0], self.max_trajectories, output.device)
            if self.mode == "fp":
                self.fp_samples[key] = output.detach().index_select(0, indices)
                return
            if self.mode != "w4a6" or not module.weight_quant:
                return

            self.encountered_quantized_modules.add(name)
            if key not in self.fp_samples:
                raise RuntimeError(f"Missing paired FP trajectory output for {key}")
            if name not in self.quant_weights:
                raise RuntimeError(f"Missing live Q(W + Win) for {name}")
            raw_input = self.raw_inputs.pop(name)
            if module.act_quant and not module.disable_act_quant:
                if name not in self.quant_inputs:
                    raise RuntimeError(f"Missing exact Q(X) for {name}")
                q_input = self.quant_inputs.pop(name).reshape(raw_input.shape)
            else:
                q_input = raw_input
            if q_input.shape[:2] != output.shape[:2]:
                raise RuntimeError(
                    f"Temporal input/output grid mismatch at {name}: "
                    f"{tuple(q_input.shape)} vs {tuple(output.shape)}"
                )

            qweight_win = self.quant_weights.pop(name)
            qweight_raw, wout = self._counterfactual_weights(name, module)
            dtype = q_input.dtype
            bias = module.bias.to(dtype) if module.bias is not None else None
            raw_quant_full = F.linear(q_input, qweight_raw.to(dtype), bias)
            base_full = F.linear(q_input, qweight_win.to(dtype), bias)
            main_full = F.linear(q_input, (qweight_win + wout).to(dtype), bias)
            ideal_u_full = F.linear(q_input, wout.to(dtype))
            actual_u_full = main_full - base_full
            masked_full = ideal_u_full * module.mask.to(ideal_u_full.dtype)
            recomputed_full = main_full + masked_full
            local_fp_full = F.linear(
                raw_input,
                module.weight.to(raw_input.dtype),
                module.bias.to(raw_input.dtype) if module.bias is not None else None,
            )

            def sample(tensor):
                return tensor.detach().index_select(0, indices).float()

            trajectory_fp = self.fp_samples[key].float()
            raw_quant = sample(raw_quant_full)
            base = sample(base_full)
            actual_u = sample(actual_u_full)
            ideal_u = sample(ideal_u_full)
            masked = sample(masked_full)
            actual = sample(output)
            recomputed = sample(recomputed_full)
            local_fp = sample(local_fp_full)
            q_sample = sample(q_input)
            reconstruction_error = relative_l2(recomputed, actual)

            common = {
                "prompt_index": self.prompt_index,
                "progress": self.progress,
                "phase": self.phase,
                "original_timestep": self.original_timestep,
                "branch": self.branch,
                "block": block_index(name),
                "role": temporal_role(name),
                "module": name,
                "sampled_trajectories": int(indices.numel()),
                "temporal_tokens": int(output.shape[1]),
                "input_channels": int(q_input.shape[-1]),
                "output_channels": int(output.shape[-1]),
                "weight_bits": int(module.weight_quantizer.n_bits),
                "act_bits": int(module.act_quantizer.n_bits),
                "actual_forward_reconstruction_relative_l2": reconstruction_error,
            }
            for reference, target in (
                ("local_same_input", local_fp),
                ("fp_trajectory", trajectory_fp),
            ):
                for component in ("dc", "ac"):
                    self.path_rows.append(
                        {
                            **common,
                            **component_record(
                                reference,
                                component,
                                target,
                                raw_quant,
                                base,
                                actual_u,
                                masked,
                                actual,
                            ),
                        }
                    )

            u_dc = project_dc(ideal_u)
            u_ac = ideal_u - u_dc
            mask = module.mask.detach().to(ideal_u.dtype)
            masked_from_dc = u_dc * mask
            masked_from_ac = u_ac * mask
            dc_to_ac = project_ac(masked_from_dc)
            ac_to_dc = project_dc(masked_from_ac)
            masked_dc = project_dc(masked)
            masked_ac = project_ac(masked)
            mask_values = module.mask.detach().float().reshape(-1)
            self.leakage_rows.append(
                {
                    **common,
                    "unmasked_dc_energy": energy(u_dc),
                    "unmasked_ac_energy": energy(u_ac),
                    "masked_dc_energy": energy(masked_dc),
                    "masked_ac_energy": energy(masked_ac),
                    "dc_to_ac_leakage_energy": energy(dc_to_ac),
                    "ac_to_dc_leakage_energy": energy(ac_to_dc),
                    "dc_to_ac_over_unmasked_dc": energy(dc_to_ac) / max(energy(u_dc), 1e-30),
                    "dc_to_ac_fraction_of_masked_ac": energy(dc_to_ac)
                    / max(energy(masked_ac), 1e-30),
                    "ac_to_dc_over_unmasked_ac": energy(ac_to_dc) / max(energy(u_ac), 1e-30),
                    "ac_to_dc_fraction_of_masked_dc": energy(ac_to_dc)
                    / max(energy(masked_dc), 1e-30),
                    "mask_mean": float(mask_values.mean()),
                    "mask_std": float(mask_values.std(unbiased=False)),
                    "mask_min": float(mask_values.min()),
                    "mask_max": float(mask_values.max()),
                }
            )

            if temporal_role(name) == "proj":
                self.rank_samples[name].append(
                    {
                        "prompt_index": self.prompt_index,
                        "progress": self.progress,
                        "phase": self.phase,
                        "branch": self.branch,
                        "block": block_index(name),
                        "x": q_sample.half().cpu(),
                        "local_same_input": (local_fp - actual).half().cpu(),
                        "fp_trajectory": (trajectory_fp - actual).half().cpu(),
                    }
                )
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()

    def release_model_refs(self):
        self.weight_cache.clear()
        self.selected_modules.clear()
        self.fp_samples.clear()
        self.qnn = None


def split_forward(qnn, x, timestep, conditioning, collector, mode, prompt, progress, phase, original_timestep):
    y = conditioning["y"]
    mask = conditioning["mask"]
    y_shape = y.shape
    y = y.reshape([2, y_shape[0] // 2] + list(y_shape[1:]))
    timestep = timestep.reshape([2, -1])
    y_cond, y_uncond = y.unbind(0)
    t_cond, t_uncond = timestep.unbind(0)
    half = x[: len(x) // 2]
    for branch, branch_y, branch_t in (
        ("conditional", y_cond, t_cond),
        ("unconditional", y_uncond, t_uncond),
    ):
        collector.begin_pass(mode, branch, prompt, progress, phase, original_timestep)
        qnn.forward(half, branch_t, branch_y, mask=mask)
        collector.end_pass()


def fit_reduced_rank_ridge(x, y, rank, args, device):
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"Bad regression shapes: X={tuple(x.shape)}, Y={tuple(y.shape)}")
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)
    q = min(x.shape[1], y.shape[1], rank + args.pca_oversample)
    if q < rank:
        raise ValueError(f"Cannot fit rank {rank} to Y={tuple(y.shape)}")
    diagonal_scale = float(torch.sum(x.square()) / max(x.shape[1], 1))
    requested_ridge = args.ridge_ratio * diagonal_scale + 1e-12
    gram = torch.matmul(x.T, x)
    rhs = torch.matmul(x.T, y)
    eye = torch.eye(x.shape[1], device=device, dtype=x.dtype)
    ridge = requested_ridge
    cholesky = None
    cholesky_info = None
    for _ in range(6):
        cholesky, cholesky_info = torch.linalg.cholesky_ex(gram + ridge * eye)
        if int(cholesky_info.max()) == 0:
            break
        ridge *= 10.0
    if cholesky is None or int(cholesky_info.max()) != 0:
        raise RuntimeError(
            f"Ridge Cholesky failed for X={tuple(x.shape)} after ridge={ridge:.6g}"
        )

    # For A=X^T X+lambda I=L L^T, ridge regression has objective
    # ||L^T(B-B*)||_F^2.  Therefore the globally optimal rank-r coefficient is
    # obtained by truncating C=L^T B*=L^{-1}X^T Y, not by taking PCA of Y.
    # This matters for a fair equal-parameter comparison: target directions
    # that X cannot predict must not make shared rank-2 look artificially weak.
    whitened_coefficient = torch.linalg.solve_triangular(
        cholesky, rhs, upper=False
    )
    left, singular, output_vectors = torch.pca_lowrank(
        whitened_coefficient, q=q, center=False, niter=args.pca_niter
    )
    output_basis = output_vectors[:, :rank].contiguous()
    whitened_input_factors = left[:, :rank] * singular[:rank].unsqueeze(0)
    input_factors = torch.linalg.solve_triangular(
        cholesky.T, whitened_input_factors, upper=True
    )
    prediction = torch.matmul(torch.matmul(x, input_factors), output_basis.T)
    train_error = energy(y - prediction)
    target_energy = energy(y)
    diagnostics = {
        "rank": rank,
        "train_rows": int(x.shape[0]),
        "input_channels": int(x.shape[1]),
        "output_channels": int(y.shape[1]),
        "requested_ridge": requested_ridge,
        "effective_ridge": ridge,
        "ridge_escalation_factor": ridge / requested_ridge,
        "train_error_energy": train_error,
        "train_target_energy": target_energy,
        "train_error_reduction": (target_energy - train_error) / max(target_energy, 1e-30),
        "leading_ridge_whitened_coefficient_singular_values": [
            float(value) for value in singular[:q]
        ],
    }
    return (input_factors, output_basis), diagnostics


def apply_model(x, model):
    input_factors, output_basis = model
    return torch.matmul(torch.matmul(x, input_factors), output_basis.T)


def stack_training(samples, reference, component=None):
    x3 = torch.cat([sample["x"] for sample in samples], dim=0).float()
    y3 = torch.cat([sample[reference] for sample in samples], dim=0).float()
    if component is not None:
        x3 = project_component(x3, component)
        y3 = project_component(y3, component)
    return x3.reshape(-1, x3.shape[-1]), y3.reshape(-1, y3.shape[-1])


def error_triplet(residual):
    return {
        "total": energy(residual),
        "dc": energy(project_dc(residual)),
        "ac": energy(project_ac(residual)),
    }


def add_errors(accumulator, prefix, values):
    for component, value in values.items():
        accumulator[f"{prefix}_{component}_error"] += value


def evaluate_frequency_models(module_name, samples, reference, models, device):
    scopes = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    for sample in samples:
        x3 = sample["x"].to(device=device, dtype=torch.float32)
        target = sample[reference].to(device=device, dtype=torch.float32)
        flat_x = x3.reshape(-1, x3.shape[-1])
        shared_rank1 = apply_model(flat_x, models["shared_rank1"]).reshape_as(target)
        shared_rank2 = apply_model(flat_x, models["shared_rank2"]).reshape_as(target)
        x_dc = project_dc(x3).reshape(-1, x3.shape[-1])
        x_ac = project_ac(x3).reshape(-1, x3.shape[-1])
        dc_prediction = apply_model(x_dc, models["dc_rank1"]).reshape_as(target)
        ac_prediction = apply_model(x_ac, models["ac_rank1"]).reshape_as(target)
        dcac_prediction = dc_prediction + ac_prediction
        errors = {
            "released": error_triplet(target),
            "shared_rank1": error_triplet(target - shared_rank1),
            "shared_rank2": error_triplet(target - shared_rank2),
            "dcac_rank1x2": error_triplet(target - dcac_prediction),
        }
        scope_keys = {
            ("overall", "all"),
            ("phase", sample["phase"]),
            ("branch", sample["branch"]),
            ("phase_branch", f'{sample["phase"]}:{sample["branch"]}'),
        }
        for scope_key in scope_keys:
            for model_name, values in errors.items():
                add_errors(scopes[scope_key], model_name, values)
            counts[scope_key] += 1

    rows = []
    for (scope, group), values in sorted(scopes.items()):
        row = {
            "module": module_name,
            "block": block_index(module_name),
            "reference": reference,
            "scope": scope,
            "group": group,
            "heldout_cells": counts[(scope, group)],
            **dict(values),
        }
        for model_name in ("shared_rank1", "shared_rank2", "dcac_rank1x2"):
            for component in ("total", "dc", "ac"):
                baseline = row[f"released_{component}_error"]
                candidate = row[f"{model_name}_{component}_error"]
                row[f"{model_name}_{component}_gain_vs_released"] = (
                    baseline - candidate
                ) / max(baseline, 1e-30)
        for component in ("total", "dc", "ac"):
            shared = row[f"shared_rank2_{component}_error"]
            dcac = row[f"dcac_rank1x2_{component}_error"]
            row[f"dcac_{component}_gain_vs_equal_param_shared_rank2"] = (
                shared - dcac
            ) / max(shared, 1e-30)
        rows.append(row)
    return rows


def run_rank_fits(rank_samples, holdout_prompt, args, device):
    fit_rows = []
    evaluation_rows = []
    for module_name, samples in sorted(rank_samples.items()):
        train = [sample for sample in samples if sample["prompt_index"] != holdout_prompt]
        heldout = [sample for sample in samples if sample["prompt_index"] == holdout_prompt]
        if not train or not heldout:
            raise RuntimeError(f"Missing train/heldout samples for {module_name}")
        for reference in ("local_same_input", "fp_trajectory"):
            torch.manual_seed(args.seed + block_index(module_name))
            full_x, full_y = stack_training(train, reference)
            dc_x, dc_y = stack_training(train, reference, "dc")
            ac_x, ac_y = stack_training(train, reference, "ac")
            models = {}
            for model_name, x, y, rank in (
                ("shared_rank1", full_x, full_y, 1),
                ("shared_rank2", full_x, full_y, 2),
                ("dc_rank1", dc_x, dc_y, 1),
                ("ac_rank1", ac_x, ac_y, 1),
            ):
                model, diagnostics = fit_reduced_rank_ridge(x, y, rank, args, device)
                models[model_name] = model
                fit_rows.append(
                    {
                        "module": module_name,
                        "block": block_index(module_name),
                        "reference": reference,
                        "fit": model_name,
                        **diagnostics,
                    }
                )
            evaluation_rows.extend(
                evaluate_frequency_models(module_name, heldout, reference, models, device)
            )
            del models, full_x, full_y, dc_x, dc_y, ac_x, ac_y
            torch.cuda.empty_cache()
    return fit_rows, evaluation_rows


def summarize_path_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["reference"], row["component"], row["role"])].append(row)
    output = []
    for key, items in sorted(grouped.items()):
        summary = {
            "reference": key[0],
            "component": key[1],
            "role": key[2],
            "rows": len(items),
        }
        for field in (
            "raw_error_energy",
            "base_after_win_error_energy",
            "unmasked_only_error_energy",
            "masked_only_error_energy",
            "released_error_energy",
        ):
            summary[field] = sum(float(row[field]) for row in items)
        for path, before, after in (
            ("win", "raw_error_energy", "base_after_win_error_energy"),
            ("unmasked", "base_after_win_error_energy", "unmasked_only_error_energy"),
            ("masked", "base_after_win_error_energy", "masked_only_error_energy"),
            ("released", "base_after_win_error_energy", "released_error_energy"),
        ):
            summary[f"aggregate_{path}_error_removal_ratio"] = (
                summary[before] - summary[after]
            ) / max(summary[before], 1e-30)
        for field in (
            "win_cosine",
            "unmasked_cosine",
            "masked_cosine",
            "total_wout_cosine",
        ):
            summary[f"mean_{field}"] = sum(float(row[field]) for row in items) / len(items)
        for field in (
            "win_overcompensates",
            "unmasked_overcompensates",
            "masked_overcompensates",
            "released_overcompensates",
        ):
            summary[f"fraction_{field}"] = sum(bool(row[field]) for row in items) / len(items)
        output.append(summary)
    return output


def summarize_leakage(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["role"]].append(row)
    output = []
    for role, items in sorted(grouped.items()):
        output.append(
            {
                "role": role,
                "rows": len(items),
                "dc_to_ac_leakage_energy": sum(row["dc_to_ac_leakage_energy"] for row in items),
                "unmasked_dc_energy": sum(row["unmasked_dc_energy"] for row in items),
                "masked_ac_energy": sum(row["masked_ac_energy"] for row in items),
                "aggregate_dc_to_ac_over_unmasked_dc": sum(
                    row["dc_to_ac_leakage_energy"] for row in items
                ) / max(sum(row["unmasked_dc_energy"] for row in items), 1e-30),
                "aggregate_dc_to_ac_fraction_of_masked_ac": sum(
                    row["dc_to_ac_leakage_energy"] for row in items
                ) / max(sum(row["masked_ac_energy"] for row in items), 1e-30),
                "mean_mask_std": sum(row["mask_std"] for row in items) / len(items),
                "mean_mask_range": sum(
                    row["mask_max"] - row["mask_min"] for row in items
                ) / len(items),
            }
        )
    return output


def aggregate_evaluation(rows, reference, scope, group):
    selected = [
        row for row in rows
        if row["reference"] == reference and row["scope"] == scope and row["group"] == group
    ]
    if not selected:
        raise RuntimeError(f"No evaluation rows for {reference}/{scope}/{group}")
    output = {"modules": len(selected)}
    for model_name in ("released", "shared_rank1", "shared_rank2", "dcac_rank1x2"):
        for component in ("total", "dc", "ac"):
            output[f"{model_name}_{component}_error"] = sum(
                row[f"{model_name}_{component}_error"] for row in selected
            )
    for model_name in ("shared_rank1", "shared_rank2", "dcac_rank1x2"):
        for component in ("total", "dc", "ac"):
            baseline = output[f"released_{component}_error"]
            candidate = output[f"{model_name}_{component}_error"]
            output[f"{model_name}_{component}_gain_vs_released"] = (
                baseline - candidate
            ) / max(baseline, 1e-30)
    for component in ("total", "dc", "ac"):
        shared = output[f"shared_rank2_{component}_error"]
        dcac = output[f"dcac_rank1x2_{component}_error"]
        output[f"dcac_{component}_gain_vs_equal_param_shared_rank2"] = (
            shared - dcac
        ) / max(shared, 1e-30)
    return output


def make_decision(evaluation_rows):
    local = aggregate_evaluation(evaluation_rows, "local_same_input", "overall", "all")
    trajectory = aggregate_evaluation(evaluation_rows, "fp_trajectory", "overall", "all")
    local_modules = [
        row for row in evaluation_rows
        if row["reference"] == "local_same_input"
        and row["scope"] == "overall"
        and row["group"] == "all"
    ]
    positive_blocks = sorted(
        row["block"]
        for row in local_modules
        if row["dcac_rank1x2_total_error"] < row["shared_rank2_total_error"]
    )
    branch_nonregression = []
    for branch in ("conditional", "unconditional"):
        values = aggregate_evaluation(evaluation_rows, "fp_trajectory", "branch", branch)
        branch_nonregression.append(
            values["dcac_rank1x2_total_error"] <= values["shared_rank2_total_error"] * 1.01
        )
    thresholds = {
        "minimum_local_total_gain_vs_equal_param_shared_rank2": 0.10,
        "minimum_local_dc_gain_vs_equal_param_shared_rank2": 0.10,
        "minimum_local_ac_gain_vs_equal_param_shared_rank2": 0.10,
        "minimum_local_ac_gain_vs_released": 0.15,
        "maximum_local_dc_regression_vs_released": 0.01,
        "minimum_positive_blocks": 3,
        "maximum_trajectory_total_regression_vs_shared_rank2": 0.01,
        "require_both_trajectory_branches_nonregressing": True,
    }
    local_dc_regression = -local["dcac_rank1x2_dc_gain_vs_released"]
    trajectory_regression_vs_shared = -trajectory[
        "dcac_total_gain_vs_equal_param_shared_rank2"
    ]
    go = (
        local["dcac_total_gain_vs_equal_param_shared_rank2"]
        >= thresholds["minimum_local_total_gain_vs_equal_param_shared_rank2"]
        and local["dcac_dc_gain_vs_equal_param_shared_rank2"]
        >= thresholds["minimum_local_dc_gain_vs_equal_param_shared_rank2"]
        and local["dcac_ac_gain_vs_equal_param_shared_rank2"]
        >= thresholds["minimum_local_ac_gain_vs_equal_param_shared_rank2"]
        and local["dcac_rank1x2_ac_gain_vs_released"]
        >= thresholds["minimum_local_ac_gain_vs_released"]
        and local_dc_regression <= thresholds["maximum_local_dc_regression_vs_released"]
        and len(positive_blocks) >= thresholds["minimum_positive_blocks"]
        and trajectory_regression_vs_shared
        <= thresholds["maximum_trajectory_total_regression_vs_shared_rank2"]
        and all(branch_nonregression)
    )
    return {
        "decision": "go" if go else "kill_or_revise",
        "meaning": (
            "Promote temporal-frequency TQE only when rank-1 DC + rank-1 AC beats an "
            "equal-parameter shared rank-2 correction on held-out data, improves AC, "
            "preserves DC, and does not damage FP-trajectory alignment."
        ),
        "local_same_input": local,
        "fp_trajectory": trajectory,
        "positive_blocks": positive_blocks,
        "trajectory_branch_nonregression": {
            "conditional": branch_nonregression[0],
            "unconditional": branch_nonregression[1],
        },
        "thresholds": thresholds,
    }


def main():
    args = parse_args()
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    selected = parse_int_set(args.selected_progress)
    blocks = parse_int_set(args.selected_blocks)
    prompt_indices = parse_int_set(args.prompt_indices)
    roles = [item.strip() for item in args.temporal_roles.split(",") if item.strip()]
    unknown = sorted(set(roles) - set(TEMPORAL_ROLES))
    if unknown:
        raise ValueError(f"Unknown temporal roles: {unknown}")
    phases, progress_to_phase = parse_phases(args.phase_groups, selected)
    if args.holdout_prompt_index not in prompt_indices:
        raise ValueError("Holdout prompt must be included in --prompt-indices")
    train_prompts = [item for item in prompt_indices if item != args.holdout_prompt_index]
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

    scheduler, latent_size, qnn = build_quant_model(args, cfg, device, dtype)
    collector = TemporalFrequencyCollector(qnn, blocks, roles, args.max_trajectories)
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
            collector.reset_progress()
            internal_timestep = total_steps - progress
            original_timestep = int(scheduler.timestep_map[internal_timestep])
            timestep = torch.tensor(
                [original_timestep] * x_t.shape[0], device=device, dtype=torch.long
            )
            phase = progress_to_phase[progress]
            set_quant_mode(qnn, False, False)
            split_forward(
                qnn, x_t, timestep, conditioning, collector, "fp", prompt_index,
                progress, phase, original_timestep,
            )
            set_quant_mode(qnn, True, True)
            split_forward(
                qnn, x_t, timestep, conditioning, collector, "w4a6", prompt_index,
                progress, phase, original_timestep,
            )
            set_quant_mode(qnn, False, False)
            assert_full_precision_state(qnn)

    collector.close()
    path_rows = collector.path_rows
    leakage_rows = collector.leakage_rows
    rank_samples = collector.rank_samples
    selected_module_count = len(collector.selected_modules)
    encountered_modules = sorted(collector.encountered_quantized_modules)
    collector.release_model_refs()
    del qnn, collector
    gc.collect()
    torch.cuda.empty_cache()

    fit_rows, evaluation_rows = run_rank_fits(
        rank_samples, args.holdout_prompt_index, args, device
    )
    path_summaries = summarize_path_rows(path_rows)
    leakage_summaries = summarize_leakage(leakage_rows)
    decision = make_decision(evaluation_rows)
    max_reconstruction_error = max(
        row["actual_forward_reconstruction_relative_l2"] for row in path_rows
    )
    if max_reconstruction_error > 5e-3:
        decision["decision"] = "invalid"
        decision["invalid_reason"] = (
            f"Recomputed released forward differs from hook output by {max_reconstruction_error:.6g}"
        )

    write_jsonl(outdir / "frequency_path_metrics.jsonl", path_rows)
    write_jsonl(outdir / "frequency_path_summaries.jsonl", path_summaries)
    write_jsonl(outdir / "mask_frequency_leakage.jsonl", leakage_rows)
    write_jsonl(outdir / "mask_frequency_leakage_summaries.jsonl", leakage_summaries)
    write_jsonl(outdir / "rank_fit_diagnostics.jsonl", fit_rows)
    write_jsonl(outdir / "heldout_equal_parameter_comparison.jsonl", evaluation_rows)
    (outdir / "decision.json").write_text(json.dumps(decision, indent=2))
    metadata = {
        "prompt_indices": prompt_indices,
        "train_prompt_indices": train_prompts,
        "holdout_prompt_index": args.holdout_prompt_index,
        "seed": args.seed,
        "selected_progress": selected,
        "phases": phases,
        "selected_blocks": blocks,
        "temporal_roles": roles,
        "primary_role": "proj",
        "auxiliary_roles": [role for role in roles if role != "proj"],
        "max_trajectories": args.max_trajectories,
        "temporal_projection_axis": 1,
        "expected_temporal_tokens": 16,
        "runtime_dtype": str(dtype),
        "quant_checkpoint": args.quant_ckpt,
        "paired_fp_trajectory": True,
        "released_forward_unchanged": True,
        "exact_activation_quantizer_output_captured": True,
        "exact_live_qweight_win_captured": True,
        "equal_parameter_control": {
            "shared": "rank-2",
            "frequency_decoupled": "rank-1 DC + rank-1 AC",
        },
        "fit_method": "ridge-whitened globally optimal reduced-rank regression",
        "ridge_ratio": args.ridge_ratio,
        "selected_module_count": selected_module_count,
        "encountered_quantized_module_count": len(encountered_modules),
        "encountered_quantized_modules": encountered_modules,
        "row_counts": {
            "frequency_path_metrics": len(path_rows),
            "frequency_path_summaries": len(path_summaries),
            "mask_frequency_leakage": len(leakage_rows),
            "mask_frequency_leakage_summaries": len(leakage_summaries),
            "rank_fit_diagnostics": len(fit_rows),
            "heldout_equal_parameter_comparison": len(evaluation_rows),
        },
        "maximum_actual_forward_reconstruction_relative_l2": max_reconstruction_error,
        "all_values_finite": all(
            math.isfinite(value)
            for rows in (
                path_rows, path_summaries, leakage_rows, leakage_summaries,
                fit_rows, evaluation_rows,
            )
            for row in rows
            for value in row.values()
            if isinstance(value, float)
        ),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"metadata": metadata, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
