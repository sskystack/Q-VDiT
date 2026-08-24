"""Paired FP/quant feature capture for Motion Transport Distillation profiling."""

import hashlib
import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F


def _extract_tensor(output):
    return output["x"] if isinstance(output, dict) else output


def _cfg_guided(cond, uncond, cfg_scale):
    """Apply the repository's exact CFG rule and return one sample per prompt."""
    cond = _extract_tensor(cond)
    uncond = _extract_tensor(uncond)
    cond_eps, cond_rest = cond[:, :3], cond[:, 3:]
    uncond_eps = uncond[:, :3]
    guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    return torch.cat([guided_eps, cond_rest], dim=1)


def _quant_state_snapshot(model):
    states = []
    for module in model.modules():
        if hasattr(module, "get_quant_state") and hasattr(module, "set_quant_state"):
            states.append((module, tuple(module.get_quant_state())))
    return states


def _restore_quant_states(states):
    for module, state in states:
        module.set_quant_state(*state)


def _safe_name(value, max_length=120):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = value.strip("_") or "prompt"
    if len(value) > max_length:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
        prefix = value[: max_length - len(digest) - 1].rstrip("_")
        value = f"{prefix}_{digest}"
    return value


def _cpu_feature(tensor):
    return tensor.detach().to(dtype=torch.float16, device="cpu").contiguous()


def capture_paired_cfg_outputs(
    model,
    latent,
    t_cond,
    t_uncond,
    y_cond,
    y_uncond,
    model_kwargs,
    quant_cond,
    quant_uncond,
    cfg_scale,
):
    """Capture paired outputs on the same latent without changing formal sampling."""
    progress = int(getattr(model, "mtd_profile_sampling_progress", -1))
    selected = set(getattr(model, "mtd_profile_steps", set()))
    if progress not in selected:
        return

    states = _quant_state_snapshot(model)
    try:
        model.set_quant_state(False, False)
        fp_cond = model.forward(latent, t_cond, y_cond, **model_kwargs)
        fp_uncond = model.forward(latent, t_uncond, y_uncond, **model_kwargs)
    finally:
        _restore_quant_states(states)

    quant_cond_tensor = _extract_tensor(quant_cond)
    fp_cond_tensor = _extract_tensor(fp_cond)
    quant_guided = _cfg_guided(quant_cond, quant_uncond, cfg_scale)
    fp_guided = _cfg_guided(fp_cond, fp_uncond, cfg_scale)
    size = int(getattr(model, "mtd_profile_transport_size", 16))
    root = Path(model.mtd_profile_dir)
    root.mkdir(parents=True, exist_ok=True)
    names = list(getattr(model, "mtd_profile_prompt_names", []))
    indices = list(getattr(model, "mtd_profile_prompt_indices", []))

    for batch_index in range(latent.shape[0]):
        prompt_name = names[batch_index] if batch_index < len(names) else str(batch_index)
        prompt_index = indices[batch_index] if batch_index < len(indices) else batch_index
        prompt_dir = root / f"{int(prompt_index):03d}_{_safe_name(prompt_name)}"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        def select_first_four(tensor):
            return tensor[batch_index : batch_index + 1, :4].float()

        payload = {
            "prompt": prompt_name,
            "prompt_index": int(prompt_index),
            "sampling_progress": progress,
            "internal_timestep": int(getattr(model, "mtd_profile_internal_timestep", -1)),
            "model_timestep": int(t_cond[batch_index].item()),
            "transport_size": size,
            "latent": _cpu_feature(latent[batch_index : batch_index + 1, :4]),
        }
        for key, tensor in {
            "quant_cond": quant_cond_tensor,
            "fp_cond": fp_cond_tensor,
            "quant_guided": quant_guided,
            "fp_guided": fp_guided,
        }.items():
            feature = select_first_four(tensor)
            pooled = F.adaptive_avg_pool2d(
                feature.permute(0, 2, 1, 3, 4).reshape(
                    feature.shape[0] * feature.shape[2],
                    feature.shape[1],
                    feature.shape[3],
                    feature.shape[4],
                ),
                (size, size),
            ).reshape(
                feature.shape[0], feature.shape[2], feature.shape[1], size, size
            ).permute(0, 2, 1, 3, 4)
            payload[key] = _cpu_feature(feature)
            payload[f"{key}_pooled"] = _cpu_feature(pooled)

        output_path = prompt_dir / f"step_{progress:03d}.pt"
        torch.save(payload, output_path)
        metadata = {
            key: value for key, value in payload.items()
            if not torch.is_tensor(value)
        }
        (prompt_dir / f"step_{progress:03d}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )
