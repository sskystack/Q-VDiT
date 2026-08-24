#!/usr/bin/env python3
"""Decompose the released Q-VDiT TQE implementation on paired FP inputs.

The released implementation contains three conceptually distinct effects:

  Cin = Q(X) [Q(W + Win) - Q(W)]^T
  Cu  = the unmasked output-side Wout effect
  Cm  = mask * Q(X) Wout^T                (temporal layers only)

Temporal layers use both Cu and Cm, whereas paper Eq. (8) describes only the
masked output-side correction.  This profiler preserves the released forward,
measures every path against the same FP layer output, and fits two-path temporal
counterfactuals on train prompts before evaluating them on a held-out prompt.
"""

import argparse
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


OPERATORS = (
    "spatial_attention",
    "temporal_attention",
    "cross_attention",
    "ffn",
)


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
    parser.add_argument("--operators", default=",".join(OPERATORS))
    parser.add_argument("--max-tokens", type=int, default=128)
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


def classify_operator(name):
    if ".attn_temp." in name:
        return "temporal_attention"
    if ".cross_attn." in name:
        return "cross_attention"
    if ".attn." in name:
        return "spatial_attention"
    if ".mlp." in name:
        return "ffn"
    return None


def evenly_spaced_indices(length, count, device):
    count = min(length, count)
    if count == length:
        return torch.arange(length, device=device)
    return torch.linspace(0, length - 1, count, device=device).round().long().unique()


def dot(left, right):
    return float(torch.sum(left.double() * right.double()))


def cosine_from_dots(path_energy, target_energy, path_target_dot):
    return path_target_dot / math.sqrt(max(path_energy * target_energy, 1e-30))


def rel_l2(left, right):
    diff = left.double() - right.double()
    return float(
        torch.linalg.vector_norm(diff)
        / torch.linalg.vector_norm(right.double()).clamp_min(1e-20)
    )


def energy_error(reference, candidate):
    return dot(reference - candidate, reference - candidate)


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


class TQEPathCollector:
    def __init__(self, qnn, blocks, operators, max_tokens):
        self.qnn = qnn
        self.blocks = set(blocks)
        self.operators = set(operators)
        self.max_tokens = max_tokens
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
        self.rows = []
        self.handles = []
        self.selected_modules = {}
        self.weight_cache = {}
        self.encountered_quantized_modules = set()
        self._register()

    def _register(self):
        for name, module in self.qnn.named_modules():
            if not isinstance(module, QuantLayer):
                continue
            block = block_index(name)
            operator = classify_operator(name)
            if block not in self.blocks or operator not in self.operators:
                continue
            if module.weight.ndim != 2 or module.fwd_func is not F.linear:
                raise RuntimeError(f"Selected non-linear QuantLayer is unsupported: {name}")
            if not isinstance(module.activation_function, StraightThrough):
                raise RuntimeError(
                    f"Additive path decomposition requires identity activation at {name}, got "
                    f"{type(module.activation_function).__name__}"
                )
            if module.split != 0:
                raise RuntimeError(f"Split QuantLayer is unsupported: {name}")
            if module.smooth_quant:
                raise RuntimeError(f"SmoothQuant must be disabled for exact path decomposition: {name}")
            self.selected_modules[name] = module
            self.handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
            self.handles.append(module.register_forward_hook(self._module_hook(name)))
            if hasattr(module, "act_quantizer"):
                self.handles.append(
                    module.act_quantizer.register_forward_hook(self._act_quant_hook(name))
                )
            self.handles.append(
                module.weight_quantizer.register_forward_hook(self._weight_quant_hook(name))
            )
        if not self.selected_modules:
            raise RuntimeError("No QuantLayers matched the selected blocks/operators")

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

    def _act_quant_hook(self, name):
        def hook(module, inputs, output):
            if self.active and self.mode == "w4a6" and torch.is_tensor(output):
                self.quant_inputs[name] = output.detach()
        return hook

    def _weight_quant_hook(self, name):
        def hook(module, inputs, output):
            if (
                self.active
                and self.mode == "w4a6"
                and not self.inside_weight_recompute
                and torch.is_tensor(output)
            ):
                # Capture the exact Q(W + Win) tensor used by the released
                # forward.  Recomputing it later is not safe because some
                # reconstruction checkpoints carry quantizer state whose
                # effective value can depend on the live forward context.
                self.quant_weights[name] = output.detach()
        return hook

    def _weights(self, name, module):
        n_bits = int(module.weight_quantizer.n_bits)
        bit_idx = int(getattr(module.weight_quantizer, "bit_idx", 0))
        key = (name, n_bits, bit_idx)
        if key not in self.weight_cache:
            win = torch.matmul(module.loraB.weight, module.loraA.weight)
            wout = torch.matmul(module.loraB_out.weight, module.loraA_out.weight)
            self.inside_weight_recompute = True
            try:
                qweight_raw = module.weight_quantizer(module.weight)
            finally:
                self.inside_weight_recompute = False
            self.weight_cache[key] = (
                qweight_raw.detach(),
                wout.detach(),
            )
        return self.weight_cache[key]

    def _module_hook(self, name):
        def hook(module, inputs, output):
            if not self.active or not torch.is_tensor(output):
                return
            call_index = self.current_call.pop(name)
            key = (self.branch, name, call_index)
            output_rows = output.detach().reshape(-1, output.shape[-1])
            indices = evenly_spaced_indices(output_rows.shape[0], self.max_tokens, output.device)
            if self.mode == "fp":
                self.fp_samples[key] = {
                    "rows": output_rows.index_select(0, indices),
                    "row_count": int(output_rows.shape[0]),
                }
                return
            if self.mode != "w4a6" or not getattr(module, "weight_quant", False):
                return

            self.encountered_quantized_modules.add(name)
            if key not in self.fp_samples:
                raise RuntimeError(f"Missing paired FP output for {key}")
            fp_entry = self.fp_samples[key]
            if fp_entry["row_count"] != output_rows.shape[0]:
                raise RuntimeError(f"FP/W4A6 row mismatch at {key}")
            fp = fp_entry["rows"].float()
            actual = output_rows.index_select(0, indices).float()

            raw_input = self.raw_inputs.pop(name)
            if getattr(module, "act_quant", False) and not module.disable_act_quant:
                if name not in self.quant_inputs:
                    raise RuntimeError(f"Exact quantized activation was not captured for {name}")
                q_input = self.quant_inputs.pop(name).reshape(raw_input.shape)
            else:
                q_input = raw_input
            q_rows = q_input.reshape(-1, q_input.shape[-1])
            if q_rows.shape[0] != output_rows.shape[0]:
                raise RuntimeError(
                    f"Input/output row mismatch at {name}: {q_rows.shape[0]} vs {output_rows.shape[0]}"
                )

            if name not in self.quant_weights:
                raise RuntimeError(f"Exact Q(W + Win) output was not captured for {name}")
            qweight_win = self.quant_weights.pop(name)
            qweight_raw, wout = self._weights(name, module)
            compute_dtype = q_input.dtype
            bias = module.bias
            if bias is not None:
                bias = bias.to(compute_dtype)
            raw_full = F.linear(q_input, qweight_raw.to(compute_dtype), bias)
            base_full = F.linear(q_input, qweight_win.to(compute_dtype), bias)
            main_full = F.linear(q_input, (qweight_win + wout).to(compute_dtype), bias)
            ideal_u_full = F.linear(q_input, wout.to(compute_dtype))
            actual_u_full = main_full - base_full
            temporal = isinstance(module, QuantTemporalAttnLinear)
            if temporal:
                mask = module.mask.to(ideal_u_full.dtype)
                masked_full = ideal_u_full * mask
                recomputed_full = main_full + masked_full
            else:
                masked_full = torch.zeros_like(ideal_u_full)
                recomputed_full = main_full

            def sample(tensor):
                return tensor.detach().reshape(-1, tensor.shape[-1]).index_select(0, indices).float()

            raw = sample(raw_full)
            base = sample(base_full)
            actual_u = sample(actual_u_full)
            ideal_u = sample(ideal_u_full)
            masked = sample(masked_full)
            recomputed = sample(recomputed_full)
            win_path = base - raw
            target_raw = fp - raw
            target_base = fp - base
            released_ideal = base + actual_u + masked

            raw_error = energy_error(fp, raw)
            base_error = energy_error(fp, base)
            release_error = energy_error(fp, actual)
            release_decomposed_error = energy_error(fp, released_ideal)
            unmasked_error = energy_error(fp, base + actual_u)
            masked_error = energy_error(fp, base + masked) if temporal else None
            win_energy = dot(win_path, win_path)
            raw_target_energy = dot(target_raw, target_raw)
            base_target_energy = dot(target_base, target_base)
            u_energy = dot(actual_u, actual_u)
            m_energy = dot(masked, masked)
            u_m_dot = dot(actual_u, masked)
            target_u_dot = dot(target_base, actual_u)
            target_m_dot = dot(target_base, masked)
            total_path = actual_u + masked
            total_energy = dot(total_path, total_path)
            target_total_dot = dot(target_base, total_path)

            operator = classify_operator(name)
            record = {
                "prompt_index": self.prompt_index,
                "progress": self.progress,
                "phase": self.phase,
                "original_timestep": self.original_timestep,
                "branch": self.branch,
                "block": block_index(name),
                "operator": operator,
                "module": name,
                "temporal_two_path": temporal,
                "sampled_tokens": int(indices.numel()),
                "input_channels": int(q_input.shape[-1]),
                "output_channels": int(output.shape[-1]),
                "weight_bits": int(module.weight_quantizer.n_bits),
                "act_bits": int(module.act_quantizer.n_bits),
                "raw_qw_error_energy": raw_error,
                "base_qw_plus_win_error_energy": base_error,
                "release_actual_error_energy": release_error,
                "release_decomposed_error_energy": release_decomposed_error,
                "unmasked_only_error_energy": unmasked_error,
                "masked_only_error_energy": masked_error,
                "win_error_removal_ratio_vs_raw": (raw_error - base_error) / max(raw_error, 1e-30),
                "unmasked_error_removal_ratio_vs_base": (base_error - unmasked_error) / max(base_error, 1e-30),
                "masked_error_removal_ratio_vs_base": (
                    (base_error - masked_error) / max(base_error, 1e-30) if temporal else None
                ),
                "release_error_removal_ratio_vs_base": (base_error - release_error) / max(base_error, 1e-30),
                "win_path_energy": win_energy,
                "unmasked_path_energy": u_energy,
                "masked_path_energy": m_energy,
                "total_wout_path_energy": total_energy,
                "win_cosine_with_raw_target": cosine_from_dots(
                    win_energy, raw_target_energy, dot(win_path, target_raw)
                ),
                "unmasked_cosine_with_base_target": cosine_from_dots(
                    u_energy, base_target_energy, target_u_dot
                ),
                "masked_cosine_with_base_target": (
                    cosine_from_dots(m_energy, base_target_energy, target_m_dot)
                    if temporal else None
                ),
                "total_wout_cosine_with_base_target": cosine_from_dots(
                    total_energy, base_target_energy, target_total_dot
                ),
                "win_overcompensates": base_error > raw_error,
                "unmasked_overcompensates": unmasked_error > base_error,
                "masked_overcompensates": masked_error > base_error if temporal else None,
                "release_overcompensates_vs_base": release_error > base_error,
                "actual_forward_reconstruction_relative_l2": rel_l2(recomputed, actual),
                "decomposed_release_vs_actual_relative_l2": rel_l2(released_ideal, actual),
                "unmasked_main_vs_ideal_relative_l2": rel_l2(actual_u, ideal_u),
                # Sufficient statistics for held-out counterfactual evaluation.
                "target_energy": base_target_energy,
                "u_energy": u_energy,
                "m_energy": m_energy,
                "u_m_dot": u_m_dot,
                "target_u_dot": target_u_dot,
                "target_m_dot": target_m_dot,
            }
            self.rows.append(record)
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


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
        collector.begin_pass(
            mode, branch, prompt, progress, phase, original_timestep
        )
        qnn.forward(half, branch_t, branch_y, mask=mask)
        collector.end_pass()


STAT_FIELDS = (
    "target_energy",
    "u_energy",
    "m_energy",
    "u_m_dot",
    "target_u_dot",
    "target_m_dot",
)


def sum_stats(rows):
    return {field: sum(float(row[field]) for row in rows) for field in STAT_FIELDS}


def temporal_error(stats, a, b):
    return (
        stats["target_energy"]
        - 2 * a * stats["target_u_dot"]
        - 2 * b * stats["target_m_dot"]
        + a * a * stats["u_energy"]
        + b * b * stats["m_energy"]
        + 2 * a * b * stats["u_m_dot"]
    )


def single_error(stats, a):
    return (
        stats["target_energy"]
        - 2 * a * stats["target_u_dot"]
        + a * a * stats["u_energy"]
    )


def fit_temporal(rows):
    stats = sum_stats(rows)
    gram = torch.tensor(
        [[stats["u_energy"], stats["u_m_dot"]],
         [stats["u_m_dot"], stats["m_energy"]]],
        dtype=torch.float64,
    )
    rhs = torch.tensor(
        [stats["target_u_dot"], stats["target_m_dot"]], dtype=torch.float64
    )
    eigen = torch.linalg.eigvalsh(gram)
    positive = eigen[eigen > max(float(eigen.max()) * 1e-12, 1e-30)]
    condition = float(eigen.max() / positive.min()) if positive.numel() else float("inf")
    ridge = float(torch.trace(gram)) * 1e-12 + 1e-24
    coeff = torch.linalg.solve(gram + ridge * torch.eye(2, dtype=torch.float64), rhs)
    return float(coeff[0]), float(coeff[1]), condition


def fit_single(rows):
    stats = sum_stats(rows)
    denominator = max(stats["u_energy"], 1e-30)
    return stats["target_u_dot"] / denominator


def group_key(row, scope):
    if scope == "block_branch":
        return (row["block"], row["branch"])
    if scope == "module_branch":
        return (row["module"], row["branch"])
    if scope == "operator_block_branch":
        return (row["operator"], row["block"], row["branch"])
    raise ValueError(scope)


def fit_and_evaluate_temporal(rows, holdout_prompt, phases):
    train = [r for r in rows if r["prompt_index"] != holdout_prompt and r["temporal_two_path"]]
    holdout = [r for r in rows if r["prompt_index"] == holdout_prompt and r["temporal_two_path"]]
    output = []
    for scope in ("block_branch", "module_branch"):
        train_groups = defaultdict(list)
        holdout_groups = defaultdict(list)
        for row in train:
            train_groups[group_key(row, scope)].append(row)
        for row in holdout:
            holdout_groups[group_key(row, scope)].append(row)
        for key, held_rows in sorted(holdout_groups.items(), key=lambda item: str(item[0])):
            if key not in train_groups:
                continue
            shared_a, shared_b, shared_condition = fit_temporal(train_groups[key])
            phase_coefficients = {}
            for phase in phases:
                phase_train = [r for r in train_groups[key] if r["phase"] == phase]
                phase_coefficients[phase] = fit_temporal(phase_train)
            for phase in list(phases) + ["__all__"]:
                evaluation = held_rows if phase == "__all__" else [r for r in held_rows if r["phase"] == phase]
                if not evaluation:
                    continue
                stats = sum_stats(evaluation)
                if phase == "__all__":
                    phase_error = sum(
                        temporal_error(sum_stats([row]), *phase_coefficients[row["phase"]][:2])
                        for row in evaluation
                    )
                    max_phase_condition = max(value[2] for value in phase_coefficients.values())
                    phase_a = None
                    phase_b = None
                else:
                    phase_a, phase_b, max_phase_condition = phase_coefficients[phase]
                    phase_error = temporal_error(stats, phase_a, phase_b)
                released_error = temporal_error(stats, 1.0, 1.0)
                output.append(
                    {
                        "scope": scope,
                        "group": list(key),
                        "phase": phase,
                        "heldout_rows": len(evaluation),
                        "released_error": released_error,
                        "paper_like_masked_only_error": temporal_error(stats, 0.0, 1.0),
                        "unmasked_only_error": temporal_error(stats, 1.0, 0.0),
                        "no_wout_error": temporal_error(stats, 0.0, 0.0),
                        "best_shared_error": temporal_error(stats, shared_a, shared_b),
                        "best_phase_specific_error": phase_error,
                        "best_shared_a": shared_a,
                        "best_shared_b": shared_b,
                        "best_phase_a": phase_a,
                        "best_phase_b": phase_b,
                        "shared_condition_number": shared_condition,
                        "phase_condition_number": max_phase_condition,
                        "shared_gain_vs_released": (
                            released_error - temporal_error(stats, shared_a, shared_b)
                        ) / max(released_error, 1e-30),
                        "phase_gain_vs_released": (released_error - phase_error)
                        / max(released_error, 1e-30),
                        "phase_gain_vs_shared": (
                            temporal_error(stats, shared_a, shared_b) - phase_error
                        ) / max(temporal_error(stats, shared_a, shared_b), 1e-30),
                    }
                )
    return output


def fit_and_evaluate_single(rows, holdout_prompt, phases):
    train = [r for r in rows if r["prompt_index"] != holdout_prompt and not r["temporal_two_path"]]
    holdout = [r for r in rows if r["prompt_index"] == holdout_prompt and not r["temporal_two_path"]]
    output = []
    for scope in ("operator_block_branch", "module_branch"):
        train_groups = defaultdict(list)
        holdout_groups = defaultdict(list)
        for row in train:
            train_groups[group_key(row, scope)].append(row)
        for row in holdout:
            holdout_groups[group_key(row, scope)].append(row)
        for key, held_rows in sorted(holdout_groups.items(), key=lambda item: str(item[0])):
            if key not in train_groups:
                continue
            shared_a = fit_single(train_groups[key])
            phase_a = {
                phase: fit_single([r for r in train_groups[key] if r["phase"] == phase])
                for phase in phases
            }
            for phase in list(phases) + ["__all__"]:
                evaluation = held_rows if phase == "__all__" else [r for r in held_rows if r["phase"] == phase]
                if not evaluation:
                    continue
                stats = sum_stats(evaluation)
                if phase == "__all__":
                    phase_error = sum(
                        single_error(sum_stats([row]), phase_a[row["phase"]]) for row in evaluation
                    )
                    fitted_a = None
                else:
                    fitted_a = phase_a[phase]
                    phase_error = single_error(stats, fitted_a)
                released_error = single_error(stats, 1.0)
                shared_error = single_error(stats, shared_a)
                output.append(
                    {
                        "scope": scope,
                        "group": list(key),
                        "phase": phase,
                        "heldout_rows": len(evaluation),
                        "released_error": released_error,
                        "no_wout_error": single_error(stats, 0.0),
                        "best_shared_error": shared_error,
                        "best_phase_specific_error": phase_error,
                        "best_shared_a": shared_a,
                        "best_phase_a": fitted_a,
                        "shared_gain_vs_released": (released_error - shared_error)
                        / max(released_error, 1e-30),
                        "phase_gain_vs_released": (released_error - phase_error)
                        / max(released_error, 1e-30),
                        "phase_gain_vs_shared": (shared_error - phase_error)
                        / max(shared_error, 1e-30),
                    }
                )
    return output


def summarize_paths(rows):
    summaries = []
    keys = (
        "raw_qw_error_energy",
        "base_qw_plus_win_error_energy",
        "release_actual_error_energy",
        "unmasked_only_error_energy",
        "masked_only_error_energy",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["phase"], row["operator"], row["block"], row["branch"])].append(row)
    for group, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        summary = {
            "phase": group[0],
            "operator": group[1],
            "block": group[2],
            "branch": group[3],
            "rows": len(items),
        }
        for key in keys:
            values = [float(row[key]) for row in items if row[key] is not None]
            summary[key] = sum(values) if values else None
        for key in (
            "win_error_removal_ratio_vs_raw",
            "unmasked_error_removal_ratio_vs_base",
            "masked_error_removal_ratio_vs_base",
            "release_error_removal_ratio_vs_base",
            "win_cosine_with_raw_target",
            "unmasked_cosine_with_base_target",
            "masked_cosine_with_base_target",
            "total_wout_cosine_with_base_target",
            "actual_forward_reconstruction_relative_l2",
            "unmasked_main_vs_ideal_relative_l2",
        ):
            values = [float(row[key]) for row in items if row[key] is not None]
            summary[f"mean_{key}"] = sum(values) / len(values) if values else None
        for key in (
            "win_overcompensates",
            "unmasked_overcompensates",
            "masked_overcompensates",
            "release_overcompensates_vs_base",
        ):
            values = [bool(row[key]) for row in items if row[key] is not None]
            summary[f"fraction_{key}"] = sum(values) / len(values) if values else None
        summaries.append(summary)
    return summaries


def temporal_decision(counterfactual_rows):
    primary = [
        row for row in counterfactual_rows
        if row["scope"] == "block_branch" and row["phase"] != "__all__"
    ]
    aggregate = [
        row for row in counterfactual_rows
        if row["scope"] == "block_branch" and row["phase"] == "__all__"
    ]
    released = sum(row["released_error"] for row in aggregate)
    phase_error = sum(row["best_phase_specific_error"] for row in aggregate)
    overall_gain = (released - phase_error) / max(released, 1e-30)
    nonregression = [
        row["best_phase_specific_error"] <= row["released_error"] * 1.001 for row in primary
    ]
    positive_blocks = set()
    for row in primary:
        if row["phase_gain_vs_released"] > 0:
            positive_blocks.add(int(row["group"][0]))
    finite_conditions = [
        row["phase_condition_number"] for row in primary
        if math.isfinite(row["phase_condition_number"])
    ]
    max_condition = max(finite_conditions) if finite_conditions else float("inf")
    thresholds = {
        "minimum_heldout_relative_error_reduction": 0.15,
        "minimum_positive_blocks": 2,
        "minimum_phase_cell_nonregression_fraction": 1.0,
        "maximum_condition_number": 1e8,
    }
    nonregression_fraction = sum(nonregression) / max(len(nonregression), 1)
    go = (
        overall_gain >= thresholds["minimum_heldout_relative_error_reduction"]
        and len(positive_blocks) >= thresholds["minimum_positive_blocks"]
        and nonregression_fraction >= thresholds["minimum_phase_cell_nonregression_fraction"]
        and max_condition <= thresholds["maximum_condition_number"]
    )
    return {
        "decision": "go" if go else "kill_or_revise",
        "meaning": (
            "This gate tests whether the released temporal Wout duplicate paths expose stable, "
            "stage-dependent held-out headroom. It does not by itself validate a rotation module."
        ),
        "heldout_released_error": released,
        "heldout_phase_specific_error": phase_error,
        "heldout_relative_error_reduction": overall_gain,
        "positive_blocks": sorted(positive_blocks),
        "phase_cell_nonregression_fraction": nonregression_fraction,
        "maximum_phase_fit_condition_number": max_condition,
        "thresholds": thresholds,
    }


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
        raise ValueError(f"Unknown operators: {unknown}")
    phases, progress_to_phase = parse_phases(args.phase_groups, selected)
    if args.holdout_prompt_index not in prompt_indices:
        raise ValueError("Holdout prompt must be included in --prompt-indices")
    train_prompts = [p for p in prompt_indices if p != args.holdout_prompt_index]
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
    collector = TQEPathCollector(qnn, blocks, operators, args.max_tokens)
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
                qnn, x_t, timestep, conditioning, collector, "fp",
                prompt_index, progress, phase, original_timestep,
            )
            set_quant_mode(qnn, True, True)
            split_forward(
                qnn, x_t, timestep, conditioning, collector, "w4a6",
                prompt_index, progress, phase, original_timestep,
            )
            set_quant_mode(qnn, False, False)
            assert_full_precision_state(qnn)
    collector.close()

    path_summaries = summarize_paths(collector.rows)
    temporal_counterfactuals = fit_and_evaluate_temporal(
        collector.rows, args.holdout_prompt_index, phases
    )
    single_counterfactuals = fit_and_evaluate_single(
        collector.rows, args.holdout_prompt_index, phases
    )
    decision = temporal_decision(temporal_counterfactuals)
    max_reconstruction_error = max(
        row["actual_forward_reconstruction_relative_l2"] for row in collector.rows
    )
    if max_reconstruction_error > 5e-3:
        decision["decision"] = "invalid"
        decision["invalid_reason"] = (
            f"Recomputed released forward differs from hook output by {max_reconstruction_error:.6g}"
        )

    write_jsonl(outdir / "path_metrics.jsonl", collector.rows)
    write_jsonl(outdir / "path_summaries.jsonl", path_summaries)
    write_jsonl(outdir / "heldout_temporal_counterfactuals.jsonl", temporal_counterfactuals)
    write_jsonl(outdir / "heldout_single_path_counterfactuals.jsonl", single_counterfactuals)
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
        "runtime_dtype": str(dtype),
        "quant_checkpoint": args.quant_ckpt,
        "paired_fp_trajectory": True,
        "released_forward_unchanged": True,
        "exact_activation_quantizer_output_captured": True,
        "selected_module_count": len(collector.selected_modules),
        "encountered_quantized_module_count": len(collector.encountered_quantized_modules),
        "encountered_quantized_modules": sorted(collector.encountered_quantized_modules),
        "row_counts": {
            "path_metrics": len(collector.rows),
            "path_summaries": len(path_summaries),
            "temporal_counterfactuals": len(temporal_counterfactuals),
            "single_path_counterfactuals": len(single_counterfactuals),
        },
        "maximum_actual_forward_reconstruction_relative_l2": max_reconstruction_error,
        "all_values_finite": all(
            math.isfinite(value)
            for rows in (
                collector.rows,
                path_summaries,
                temporal_counterfactuals,
                single_counterfactuals,
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
