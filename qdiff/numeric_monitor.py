import json
import math
import os
import time

import torch


FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0 ** -14
FP16_MIN_SUBNORMAL = 2.0 ** -24


def _as_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def tensor_stats(tensor):
    """Return numerical-range statistics without copying the full tensor to CPU."""
    if tensor is None:
        return None
    value = tensor.detach()
    count = value.numel()
    if count == 0:
        return {"numel": 0, "dtype": str(value.dtype), "device": str(value.device)}

    finite = torch.isfinite(value)
    finite_count = int(finite.sum().item())
    nonfinite_count = count - finite_count
    result = {
        "numel": int(count),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "nonfinite_count": int(nonfinite_count),
        "nonfinite_ratio": float(nonfinite_count / count),
    }
    if finite_count == 0:
        return result

    finite_value = value[finite]
    finite_float = finite_value.float()
    abs_value = finite_float.abs()
    nonzero = abs_value > 0
    result.update({
        "min": float(finite_float.min().item()),
        "max": float(finite_float.max().item()),
        "mean": float(finite_float.mean().item()),
        "std": float(finite_float.std(unbiased=False).item()),
        "abs_max": float(abs_value.max().item()),
        "abs_mean": float(abs_value.mean().item()),
        "zero_count": int((~nonzero).sum().item()),
        "zero_ratio": float((~nonzero).float().mean().item()),
        "fp16_subnormal_count": int(
            (nonzero & (abs_value < FP16_MIN_NORMAL)).sum().item()
        ),
        "fp16_subnormal_ratio": float(
            (nonzero & (abs_value < FP16_MIN_NORMAL)).float().mean().item()
        ),
        "below_fp16_min_subnormal_count": int(
            (nonzero & (abs_value < FP16_MIN_SUBNORMAL)).sum().item()
        ),
        "below_fp16_min_subnormal_ratio": float(
            (nonzero & (abs_value < FP16_MIN_SUBNORMAL)).float().mean().item()
        ),
        "over_fp16_range_count": int((abs_value > FP16_MAX).sum().item()),
        "over_fp16_range_ratio": float(
            (abs_value > FP16_MAX).float().mean().item()
        ),
    })
    return result


def cast_recovery_stats(gradient, dtype, scale=1.0):
    """Measure information lost when a FP32 gradient crosses a low-precision cast."""
    value = gradient.detach().float()
    scale = float(scale)
    recovered = (value * scale).to(dtype).float() / scale
    difference = value - recovered
    value_norm = torch.linalg.vector_norm(value)
    recovered_norm = torch.linalg.vector_norm(recovered)
    difference_norm = torch.linalg.vector_norm(difference)
    cosine = (
        (value.flatten() @ recovered.flatten())
        / (value_norm * recovered_norm + 1.0e-30)
    )
    return {
        "dtype": str(dtype),
        "scale": scale,
        "zero_count": int((recovered == 0).sum().item()),
        "zero_ratio": float((recovered == 0).float().mean().item()),
        "original_l2_norm": float(value_norm.item()),
        "recovered_l2_norm": float(recovered_norm.item()),
        "relative_l2_error": float(
            (difference_norm / (value_norm + 1.0e-30)).item()
        ),
        "cosine": float(cosine.item()),
    }


def mean_mse_output_gradient_stats(residual, active_loss_scale=1.0):
    """Profile d(mean-MSE)/d(output) before it re-enters the model backward."""
    gradient = 2.0 * residual.detach().float() / residual.numel()
    return {
        "gradient": tensor_stats(gradient),
        "fp16_unscaled_cast": cast_recovery_stats(
            gradient, torch.float16, scale=1.0
        ),
        "fp16_active_scale_cast": cast_recovery_stats(
            gradient, torch.float16, scale=active_loss_scale
        ),
        "fp16_scale_4096_cast": cast_recovery_stats(
            gradient, torch.float16, scale=4096.0
        ),
        "bf16_unscaled_cast": cast_recovery_stats(
            gradient, torch.bfloat16, scale=1.0
        ),
    }


def parameter_category(name):
    lowered = name.lower()
    if "delta" in lowered:
        return "delta"
    if "lora" in lowered:
        return "lora"
    if "mask" in lowered:
        return "mask"
    return "other"


class NumericMonitor:
    def __init__(self, output_path, interval=25, detailed_interval=100):
        self.output_path = output_path
        self.interval = max(1, int(interval))
        self.detailed_interval = max(self.interval, int(detailed_interval))
        self.records_written = 0
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def should_record(self, iteration):
        return self.records_written == 0 or iteration % self.interval == 0

    def should_record_details(self, iteration):
        return self.records_written == 0 or iteration % self.detailed_interval == 0

    def _write(self, record):
        record = dict(record)
        record["wall_time"] = time.time()
        with open(self.output_path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    _json_safe(record), ensure_ascii=False, allow_nan=False
                ) + "\n"
            )

    def record_iteration(
        self,
        iteration,
        module,
        optimizer,
        output,
        target,
        loss,
        loss_components=None,
        scaler=None,
        learning_rates=None,
    ):
        if not self.should_record(iteration):
            return
        write_details = self.should_record_details(iteration)

        trainable = [
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ]
        category_summary = {}
        for category in ("delta", "lora", "mask", "other"):
            members = [
                (name, parameter)
                for name, parameter in trainable
                if parameter_category(name) == category
            ]
            if not members:
                continue
            total = sum(parameter.numel() for _, parameter in members)
            grad_total = sum(
                parameter.grad.numel()
                for _, parameter in members
                if parameter.grad is not None
            )
            grad_zero = sum(
                int((parameter.grad.detach() == 0).sum().item())
                for _, parameter in members
                if parameter.grad is not None
            )
            grad_nonfinite = sum(
                int((~torch.isfinite(parameter.grad.detach())).sum().item())
                for _, parameter in members
                if parameter.grad is not None
            )
            grad_sq_sum = 0.0
            for _, parameter in members:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                if torch.isfinite(gradient).all():
                    grad_sq_sum += float(
                        gradient.float().square().sum().item()
                    )
            category_summary[category] = {
                "parameter_tensors": len(members),
                "parameter_numel": int(total),
                "gradient_numel": int(grad_total),
                "gradient_zero_ratio": (
                    float(grad_zero / grad_total) if grad_total else None
                ),
                "gradient_nonfinite_count": int(grad_nonfinite),
                "gradient_l2_norm": (
                    math.sqrt(grad_sq_sum) if grad_nonfinite == 0 else None
                ),
            }

        residual = output.detach().float() - target.detach().float()
        active_loss_scale = (
            float(scaler.get_scale())
            if scaler is not None and scaler.is_enabled()
            else 1.0
        )
        record = {
            "kind": "iteration",
            "iteration": int(iteration),
            "loss": _as_float(loss),
            "loss_components": loss_components or {},
            "learning_rates": learning_rates or [
                float(group["lr"]) for group in optimizer.param_groups
            ],
            "grad_scaler_scale": (
                active_loss_scale
                if scaler is not None and scaler.is_enabled()
                else None
            ),
            "output": tensor_stats(output),
            "target": tensor_stats(target),
            "residual": tensor_stats(residual),
            "mean_mse_output_gradient": (
                mean_mse_output_gradient_stats(
                    residual, active_loss_scale=active_loss_scale
                )
                if write_details
                else None
            ),
            "trainable_categories": category_summary,
        }
        self._write(record)
        self.records_written += 1

        if write_details:
            for name, parameter in trainable:
                state = optimizer.state.get(parameter, {})
                self._write({
                    "kind": "parameter",
                    "iteration": int(iteration),
                    "name": name,
                    "category": parameter_category(name),
                    "parameter": tensor_stats(parameter),
                    "gradient": tensor_stats(parameter.grad),
                    "optimizer_exp_avg": tensor_stats(state.get("exp_avg")),
                    "optimizer_exp_avg_sq": tensor_stats(state.get("exp_avg_sq")),
                    "optimizer_step": _as_float(state.get("step")),
                })
