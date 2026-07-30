"""Paired numerical tracing for FP16/BF16 PTQ calibration runs.

The profiler is intentionally read-only: it installs forward hooks, samples a
small number of token rows, and writes JSONL records.  It never changes model
parameters, quantizer state, gradients, or forward outputs.
"""

import json
import math
import os
import time

import torch
import torch.nn.functional as F

from qdiff.numeric_monitor import FP16_MAX, tensor_stats


def _safe_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _relative_rmse(value, reference):
    value = value.detach().float()
    reference = reference.detach().float()
    error_rms = (value - reference).square().mean().sqrt()
    reference_rms = reference.square().mean().sqrt()
    return _safe_number(error_rms / reference_rms.clamp_min(1.0e-12))


def _cast_stats(value, dtype):
    source = value.detach().float()
    casted = source.to(dtype).float()
    nonzero = source != 0
    nonzero_count = int(nonzero.sum().item())
    zeroed = nonzero & (casted == 0)
    return {
        "target_dtype": str(dtype),
        "relative_rmse": _relative_rmse(casted, source),
        "nonzero_to_zero_count": int(zeroed.sum().item()),
        "nonzero_to_zero_ratio": (
            float(zeroed.sum().item() / nonzero_count) if nonzero_count else 0.0
        ),
        "source_over_fp16_range_count": int(
            (source.abs() > FP16_MAX).sum().item()
        ),
        "source_over_fp16_range_ratio": float(
            (source.abs() > FP16_MAX).float().mean().item()
        ),
    }


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _compact_tensor_stats(value, sample_limit=131072):
    """Avoid multi-gigabyte FP32 copies for STDiT hidden-state statistics."""
    if value is None or value.numel() <= 1_000_000:
        return tensor_stats(value)
    detached = value.detach()
    flat = detached.reshape(-1)
    count = min(int(sample_limit), flat.numel())
    if count == 1:
        indices = torch.zeros(1, device=flat.device, dtype=torch.long)
    else:
        indices = (
            torch.arange(count, device=flat.device, dtype=torch.long)
            * (flat.numel() - 1)
            // (count - 1)
        )
    sample = flat.index_select(0, indices)
    stats = tensor_stats(sample)
    stats.update({
        "numel": int(detached.numel()),
        "sample_numel": int(sample.numel()),
        "sampled_moments": True,
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "exact_min": _safe_number(detached.min()),
        "exact_max": _safe_number(detached.max()),
        "exact_abs_max": _safe_number(detached.abs().max()),
        "all_finite": bool(torch.isfinite(detached).all().item()),
    })
    return stats


class CalibNumericProfiler:
    def __init__(self, model, output_path, max_rows=32):
        self.model = model
        self.output_path = output_path
        self.max_rows = max(1, int(max_rows))
        self.phase = "setup"
        self.forward_index = 0
        self.current_timesteps = []
        self.in_full_forward = False
        self.seen = set()
        self.handles = []
        self.module_names = {id(module): name for name, module in model.named_modules()}
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self._install_hooks()
        self._write({
            "kind": "profiler_start",
            "max_rows": self.max_rows,
            "torch_version": torch.__version__,
        })

    def _write(self, record):
        payload = dict(record)
        payload.update({
            "phase": self.phase,
            "forward_index": self.forward_index,
            "timesteps": self.current_timesteps,
            "wall_time": time.time(),
        })
        with open(self.output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")

    def set_phase(self, phase):
        self.phase = str(phase)
        self._write({"kind": "phase"})

    def _time_bucket(self):
        if not self.current_timesteps:
            return "unknown"
        timestep = max(self.current_timesteps)
        if timestep >= 900:
            return "t900_999"
        if timestep >= 700:
            return "t700_899"
        if timestep >= 400:
            return "t400_699"
        if timestep >= 100:
            return "t100_399"
        return "t000_099"

    def _claim(self, kind, module_name):
        key = (self.phase, self._time_bucket(), kind, module_name)
        if key in self.seen:
            return False
        self.seen.add(key)
        return True

    def _install_hooks(self):
        self.handles.append(self.model.register_forward_pre_hook(self._model_pre_hook))
        self.handles.append(self.model.register_forward_hook(self._model_post_hook))
        for name, module in self.model.named_modules():
            class_name = module.__class__.__name__
            if hasattr(module, "weight_quantizer") and hasattr(module, "fwd_func"):
                self.handles.append(module.register_forward_hook(self._quant_layer_hook))
            elif class_name == "STDiTBlock":
                self.handles.append(module.register_forward_hook(self._block_hook))

    def _model_pre_hook(self, module, inputs):
        self.forward_index += 1
        self.in_full_forward = True
        timestep = inputs[1] if len(inputs) > 1 and torch.is_tensor(inputs[1]) else None
        if timestep is None:
            self.current_timesteps = []
        else:
            values = torch.unique(timestep.detach().to("cpu")).tolist()
            self.current_timesteps = [int(value) for value in values[:16]]
        if self._claim("model_input", "model"):
            self._write({
                "kind": "model_input",
                "latent": _compact_tensor_stats(inputs[0] if inputs else None),
                "timestep_tensor": tensor_stats(timestep),
                "condition": _compact_tensor_stats(
                    inputs[2] if len(inputs) > 2 and torch.is_tensor(inputs[2]) else None
                ),
            })

    def _model_post_hook(self, module, inputs, output):
        if self._claim("model_output", "model"):
            self._write({
                "kind": "model_output",
                "output": _compact_tensor_stats(_first_tensor(output)),
            })
        self.in_full_forward = False

    def _block_hook(self, module, inputs, output):
        if not self.in_full_forward:
            return
        name = self.module_names.get(id(module), module.__class__.__name__)
        if not self._claim("block", name):
            return
        input_tensor = _first_tensor(inputs)
        output_tensor = _first_tensor(output)
        record = {
            "kind": "block",
            "module": name,
            "input": _compact_tensor_stats(input_tensor),
            "output": _compact_tensor_stats(output_tensor),
        }
        if input_tensor is not None and output_tensor is not None:
            input_sample, indices = self._sample_rows(input_tensor)
            input_rows = input_tensor.numel() // input_tensor.shape[-1]
            output_rows = output_tensor.numel() // output_tensor.shape[-1]
            if input_rows == output_rows:
                output_sample, _ = self._sample_rows(output_tensor, indices)
            else:
                output_sample, _ = self._sample_rows(output_tensor)
            input_rms = input_sample.float().square().mean().sqrt()
            output_rms = output_sample.float().square().mean().sqrt()
            record["output_to_input_rms_ratio"] = _safe_number(
                output_rms / input_rms.clamp_min(1.0e-12)
            )
        self._write(record)

    def _sample_rows(self, tensor, indices=None):
        flat = tensor.detach().reshape(-1, tensor.shape[-1])
        if indices is None:
            count = min(self.max_rows, flat.shape[0])
            if count == flat.shape[0]:
                indices = torch.arange(flat.shape[0], device=flat.device)
            elif count == 1:
                indices = torch.zeros(1, device=flat.device, dtype=torch.long)
            else:
                indices = (
                    torch.arange(count, device=flat.device, dtype=torch.long)
                    * (flat.shape[0] - 1)
                    // (count - 1)
                )
        return flat.index_select(0, indices), indices

    def _quantized_activation(self, module, raw_input, row_indices):
        raw_sample, _ = self._sample_rows(raw_input, row_indices)
        if not getattr(module, "act_quant", False):
            return raw_sample, None
        if getattr(module, "disable_act_quant", False):
            return raw_sample, None
        if getattr(module, "split", 0) != 0 or getattr(module, "smooth_quant", False):
            return None, {"skipped": "split_or_smooth_quant"}
        quantizer = module.act_quantizer
        delta = getattr(quantizer, "delta", None)
        zero_point = getattr(quantizer, "zero_point", None)
        if delta is None or zero_point is None:
            return None, {"skipped": "uninitialized"}
        source = raw_input.detach()
        if source.ndim != 3 or delta.numel() != source.shape[1]:
            return None, {"skipped": "unsupported_activation_shape"}
        token_indices = row_indices.remainder(source.shape[1])
        delta_sample = delta.reshape(-1).index_select(0, token_indices).reshape(-1, 1)
        zero_sample = zero_point.reshape(-1).index_select(0, token_indices).reshape(-1, 1)
        scaled = raw_sample / delta_sample
        x_int = torch.round(scaled) + zero_sample
        if quantizer.sym:
            x_quant = torch.clamp(x_int, -quantizer.n_levels - 1, quantizer.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, quantizer.n_levels - 1)
        quantized = (
            x_quant * delta_sample
            if quantizer.sym
            else (x_quant - zero_sample) * delta_sample
        )
        saturation = (
            (x_int <= 0) | (x_int >= quantizer.n_levels - 1)
            if not quantizer.sym
            else (x_int <= -quantizer.n_levels - 1) | (x_int >= quantizer.n_levels)
        )
        return quantized, {
            "n_bits": int(quantizer.n_bits),
            "delta": tensor_stats(delta),
            "zero_point": tensor_stats(zero_point),
            "scaled_abs_max": _safe_number(scaled.detach().float().abs().max()),
            "saturation_ratio": float(saturation.float().mean().item()),
            "relative_rmse": _relative_rmse(quantized, raw_sample),
            "sample_rows": int(raw_sample.shape[0]),
        }

    def _effective_weight(self, module):
        if not getattr(module, "weight_quant", False):
            return module.weight.detach(), None, None
        if getattr(module, "split", 0) != 0 or getattr(module, "smooth_quant", False):
            return None, None, None
        lora_main = None
        lora_out = None
        weight_source = module.weight
        if all(hasattr(module, name) for name in ("loraA", "loraB", "loraA_out", "loraB_out")):
            lora_main = module.loraB.weight @ module.loraA.weight
            lora_out = module.loraB_out.weight @ module.loraA_out.weight
            weight_source = weight_source + lora_main
        # The first model forward initializes delta/zero-point, but the global
        # init_done flag is set only after that forward returns.  Calling the
        # quantizer module here would initialize it a second time and perturb
        # calibration.  Reconstruct dequantized weights directly from the
        # already-created parameters instead.
        quantizer = module.weight_quantizer
        x_quant = quantizer.rounding(weight_source)
        effective = (
            x_quant * quantizer.delta
            if quantizer.sym
            else (x_quant - quantizer.zero_point) * quantizer.delta
        )
        if lora_out is not None:
            effective = effective + lora_out
        return effective.detach(), lora_main.detach() if lora_main is not None else None, lora_out.detach() if lora_out is not None else None

    def _linear_error_decomposition(
        self, module, raw_input, quantized_input, effective_weight, actual_output
    ):
        if effective_weight is None or effective_weight.ndim != 2:
            return {"skipped": "non_linear_or_unavailable_weight"}
        if raw_input.shape[-1] != effective_weight.shape[1]:
            return {"skipped": "input_shape_mismatch"}
        raw = raw_input.float()
        quantized = quantized_input.float()
        actual = actual_output.float() if actual_output is not None else None
        base_weight = module.weight.detach().float()
        effective = effective_weight.detach().float()
        bias = module.bias.detach().float() if module.bias is not None else None
        fp_output = F.linear(raw, base_weight, bias)
        activation_only = F.linear(quantized, base_weight, bias)
        weight_only = F.linear(raw, effective, bias)
        quantized_fp32 = F.linear(quantized, effective, bias)
        return {
            "sample_rows": int(raw.shape[0]),
            "fp_output": tensor_stats(fp_output),
            "quantized_fp32_output": tensor_stats(quantized_fp32),
            "activation_only_relative_rmse": _relative_rmse(activation_only, fp_output),
            "weight_only_relative_rmse": _relative_rmse(weight_only, fp_output),
            "combined_quant_relative_rmse": _relative_rmse(quantized_fp32, fp_output),
            "runtime_compute_relative_rmse": (
                _relative_rmse(actual, quantized_fp32)
                if actual is not None and actual.shape == quantized_fp32.shape
                else None
            ),
            "quantized_fp32_over_fp16_range_ratio": float(
                (quantized_fp32.abs() > FP16_MAX).float().mean().item()
            ),
        }

    def _quant_layer_hook(self, module, inputs, output):
        if not self.in_full_forward:
            return
        name = self.module_names.get(id(module), module.__class__.__name__)
        if not self._claim("quant_layer", name):
            return
        raw_input = _first_tensor(inputs)
        actual_output = _first_tensor(output)
        if raw_input is None or actual_output is None:
            return
        with torch.no_grad():
            raw_sample, row_indices = self._sample_rows(raw_input)
            raw_rows = raw_input.numel() // raw_input.shape[-1]
            output_rows = actual_output.numel() // actual_output.shape[-1]
            rows_aligned = raw_rows == output_rows
            if rows_aligned:
                actual_sample, _ = self._sample_rows(actual_output, row_indices)
            else:
                actual_sample = None
            quantized_input, activation = self._quantized_activation(
                module, raw_input, row_indices
            )
            effective_weight, lora_main, lora_out = self._effective_weight(module)
            record = {
                "kind": "quant_layer",
                "module": name,
                "class": module.__class__.__name__,
                "weight_quant": bool(getattr(module, "weight_quant", False)),
                "act_quant": bool(getattr(module, "act_quant", False)),
                "raw_input": _compact_tensor_stats(raw_input),
                "actual_output": _compact_tensor_stats(actual_output),
                "base_weight": tensor_stats(module.weight),
                "activation_quantization": activation,
            }
            if effective_weight is not None:
                record["effective_weight"] = tensor_stats(effective_weight)
                record["effective_weight_runtime_cast"] = _cast_stats(
                    effective_weight, raw_input.dtype
                )
            if lora_main is not None:
                record["lora_main"] = tensor_stats(lora_main)
                record["lora_main_runtime_cast"] = _cast_stats(
                    lora_main, raw_input.dtype
                )
            if lora_out is not None:
                record["lora_out"] = tensor_stats(lora_out)
                record["lora_out_runtime_cast"] = _cast_stats(
                    lora_out, raw_input.dtype
                )
            if quantized_input is not None:
                record["error_decomposition"] = self._linear_error_decomposition(
                    module,
                    raw_sample,
                    quantized_input,
                    effective_weight,
                    actual_sample,
                )
                record["error_decomposition"]["rows_aligned"] = rows_aligned
            self._write(record)

    def close(self):
        self._write({"kind": "profiler_stop"})
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


_PROFILER = None


def start_calib_numeric_profiler(model, output_dir):
    global _PROFILER
    enabled = os.environ.get("QVDIT_CALIB_NUMERIC_PROFILE", "0") == "1"
    if not enabled:
        return None
    max_rows = int(os.environ.get("QVDIT_CALIB_NUMERIC_MAX_ROWS", "32"))
    output_path = os.path.join(output_dir, "calib_numeric_trace.jsonl")
    _PROFILER = CalibNumericProfiler(model, output_path, max_rows=max_rows)
    return _PROFILER


def set_calib_numeric_phase(phase):
    if _PROFILER is not None:
        _PROFILER.set_phase(phase)


def finish_calib_numeric_profiler():
    global _PROFILER
    if _PROFILER is not None:
        _PROFILER.close()
        _PROFILER = None


def record_calib_numeric_comparison(label, reference, candidate):
    if _PROFILER is None:
        return
    residual = candidate.detach().float() - reference.detach().float()
    _PROFILER._write({
        "kind": "probe_comparison",
        "label": str(label),
        "reference": _compact_tensor_stats(reference),
        "candidate": _compact_tensor_stats(candidate),
        "residual": _compact_tensor_stats(residual),
        "relative_rmse": _relative_rmse(candidate, reference),
    })
