"""Teacher-feature diagnostics for adaptive Motion Transport Distillation.

This module is deliberately separate from the released MTD training path.  It
tests whether a fixed, same-centre 3x3 correspondence set misses useful teacher
matches and whether cheap per-token signals identify positions that dominate
the current MTD error.  No optical flow or learned selector is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DiagnosticConfig:
    search_radius: int = 2
    topk: int = 9
    temperature: float = 0.07
    seed: int = 0


def _validate_features(teacher, student, config):
    if teacher.shape != student.shape:
        raise ValueError(
            f"teacher/student shapes differ: {tuple(teacher.shape)} vs "
            f"{tuple(student.shape)}"
        )
    if teacher.ndim != 5:
        raise ValueError("expected features shaped [batch, channels, frames, height, width]")
    if teacher.shape[2] < 2:
        raise ValueError("at least two frames are required")
    if config.search_radius < 1:
        raise ValueError("search_radius must be at least 1")
    if config.topk <= 0:
        raise ValueError("topk must be positive")
    if config.temperature <= 0:
        raise ValueError("temperature must be positive")


def _rankdata(values):
    """Average ranks for a one-dimensional tensor, matching Spearman ties."""
    values = values.detach().double().flatten()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(values, target):
    values = values.detach().double().flatten()
    target = target.detach().double().flatten()
    finite = torch.isfinite(values) & torch.isfinite(target)
    if finite.sum() < 2:
        return float("nan")
    left = _rankdata(values[finite])
    right = _rankdata(target[finite])
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if denominator == 0:
        return 0.0
    return float((left * right).sum() / denominator)


def _prepare(features, radius):
    batch, channels, frames, height, width = features.shape
    normalized = F.normalize(features.float(), dim=1, eps=1.0e-6)
    current = normalized[:, :, :-1]
    following = normalized[:, :, 1:]
    pairs = batch * (frames - 1)
    tokens = height * width
    kernel = 2 * radius + 1

    current_flat = current.permute(0, 2, 3, 4, 1).reshape(
        pairs, tokens, channels
    )
    following_flat = following.permute(0, 2, 1, 3, 4).reshape(
        pairs, channels, height, width
    )
    neighbours = F.unfold(
        following_flat, kernel_size=kernel, padding=radius
    ).reshape(pairs, channels, kernel * kernel, tokens).permute(0, 3, 2, 1)

    valid_grid = following_flat.new_ones(pairs, 1, height, width)
    valid = F.unfold(
        valid_grid, kernel_size=kernel, padding=radius
    ).reshape(pairs, kernel * kernel, tokens).permute(0, 2, 1).bool()

    offsets = torch.stack(
        torch.meshgrid(
            torch.arange(-radius, radius + 1, device=features.device),
            torch.arange(-radius, radius + 1, device=features.device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    return current_flat, neighbours, valid, offsets


def _masked_softmax(logits, mask):
    floor = torch.finfo(logits.dtype).min
    probabilities = F.softmax(logits.masked_fill(~mask, floor), dim=-1)
    return probabilities.masked_fill(~mask, 0.0)


def _entropy(probabilities, mask):
    entropy = -(
        probabilities.clamp_min(1.0e-12).log() * probabilities
    ).sum(dim=-1)
    count = mask.sum(dim=-1).clamp_min(1)
    normalizer = count.float().log().clamp_min(1.0)
    return entropy, entropy / normalizer


def _topk_mask(scores, valid, topk):
    k = min(topk, scores.shape[-1])
    masked = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    indices = masked.topk(k, dim=-1).indices
    selected = torch.zeros_like(valid)
    selected.scatter_(-1, indices, True)
    return selected & valid


def _random_topk_mask(valid, topk, seed):
    generator = torch.Generator(device=valid.device)
    generator.manual_seed(seed)
    scores = torch.rand(valid.shape, generator=generator, device=valid.device)
    return _topk_mask(scores, valid, topk)


def _centered_mask(valid, offsets, centre_indices):
    centres = offsets[centre_indices]
    delta = offsets.view(1, 1, -1, 2) - centres.unsqueeze(-2)
    return valid & (delta.abs().amax(dim=-1) <= 1)


def _scheme_metrics(
    teacher_current,
    teacher_neighbours,
    teacher_logits,
    student_current,
    student_neighbours,
    student_logits,
    full_teacher_probabilities,
    offsets,
    mask,
):
    teacher_probabilities = _masked_softmax(teacher_logits, mask)
    student_probabilities = _masked_softmax(student_logits, mask)
    local_kl = (
        teacher_probabilities
        * (
            teacher_probabilities.clamp_min(1.0e-8).log()
            - student_probabilities.clamp_min(1.0e-8).log()
        )
    ).sum(dim=-1)

    teacher_transport = (
        teacher_probabilities[..., None] * teacher_neighbours
    ).sum(dim=-2) - teacher_current
    student_transport = (
        student_probabilities[..., None] * student_neighbours
    ).sum(dim=-2) - student_current
    motion_error = F.smooth_l1_loss(
        student_transport, teacher_transport, reduction="none"
    ).mean(dim=-1)

    offsets_float = offsets.to(dtype=teacher_probabilities.dtype)
    teacher_displacement = (
        teacher_probabilities[..., None] * offsets_float
    ).sum(dim=-2)
    student_displacement = (
        student_probabilities[..., None] * offsets_float
    ).sum(dim=-2)
    displacement_error = (
        teacher_displacement - student_displacement
    ).square().sum(dim=-1).sqrt()

    entropy, normalized_entropy = _entropy(teacher_probabilities, mask)
    full_top1 = full_teacher_probabilities.argmax(dim=-1)
    return {
        "local_kl": local_kl,
        "motion_error": motion_error,
        "displacement_error": displacement_error,
        "teacher_entropy": entropy,
        "teacher_normalized_entropy": normalized_entropy,
        "teacher_mass_covered": (full_teacher_probabilities * mask).sum(dim=-1),
        "teacher_top1_retained": mask.gather(
            -1, full_top1.unsqueeze(-1)
        ).squeeze(-1).float(),
        "candidate_count": mask.sum(dim=-1).float(),
        "teacher_displacement": teacher_displacement,
    }


def analyze_adaptive_correspondence(teacher, student, config=None):
    """Return per-query diagnostics for fixed and adaptive candidate sets."""
    config = config or DiagnosticConfig()
    _validate_features(teacher, student, config)

    # Keep a one-cell refinement margin around the coarse search window.  This
    # lets a coarse centre at +/-search_radius use a complete local 3x3 rather
    # than being artificially truncated by the diagnostic bank itself.
    bank_radius = config.search_radius + 1
    teacher_current, teacher_neighbours, valid, offsets = _prepare(
        teacher, bank_radius
    )
    student_current, student_neighbours, student_valid, _ = _prepare(
        student, bank_radius
    )
    if not torch.equal(valid, student_valid):
        raise RuntimeError("teacher/student candidate validity differs")

    teacher_logits = (
        teacher_current[:, :, None, :] * teacher_neighbours
    ).sum(dim=-1) / config.temperature
    student_logits = (
        student_current[:, :, None, :] * student_neighbours
    ).sum(dim=-1) / config.temperature
    wide_mask = valid & (
        offsets.abs().amax(dim=-1).view(1, 1, -1)
        <= config.search_radius
    )
    full_teacher_probabilities = _masked_softmax(teacher_logits, wide_mask)

    fixed_mask = valid & (
        offsets.abs().amax(dim=-1).view(1, 1, -1) <= 1
    )
    topk_mask = _topk_mask(teacher_logits, wide_mask, config.topk)
    teacher_top1 = teacher_logits.masked_fill(
        ~wide_mask, torch.finfo(teacher_logits.dtype).min
    ).argmax(dim=-1)
    centred_mask = _centered_mask(valid, offsets, teacher_top1)
    random_mask = _random_topk_mask(wide_mask, config.topk, config.seed)

    schemes = {
        "fixed_3x3": fixed_mask,
        "teacher_topk": topk_mask,
        "teacher_centered_3x3": centred_mask,
        "random_topk": random_mask,
    }
    metrics = {
        name: _scheme_metrics(
            teacher_current,
            teacher_neighbours,
            teacher_logits,
            student_current,
            student_neighbours,
            student_logits,
            full_teacher_probabilities,
            offsets,
            mask,
        )
        for name, mask in schemes.items()
    }

    fixed = metrics["fixed_3x3"]
    full_entropy, full_normalized_entropy = _entropy(
        full_teacher_probabilities, wide_mask
    )
    zero_offset = (
        (offsets[:, 0] == 0) & (offsets[:, 1] == 0)
    ).nonzero(as_tuple=False).item()
    same_position_following = teacher_neighbours[:, :, zero_offset]

    local_mean = fixed["local_kl"].mean().clamp_min(1.0e-12)
    motion_mean = fixed["motion_error"].mean().clamp_min(1.0e-12)
    need_score = (
        fixed["local_kl"] / local_mean
        + fixed["motion_error"] / motion_mean
    )
    proxies = {
        "fixed_match_confidence": 1.0 - fixed["teacher_normalized_entropy"],
        "wide_match_confidence": 1.0 - full_normalized_entropy,
        "teacher_temporal_delta": (
            teacher_current - same_position_following
        ).square().mean(dim=-1).sqrt(),
        "teacher_motion_magnitude": metrics["teacher_topk"][
            "teacher_displacement"
        ].square().sum(dim=-1).sqrt(),
        "fp_quant_error": (
            teacher_current - student_current
        ).square().mean(dim=-1).sqrt(),
        "oracle_need_score": need_score,
    }

    fixed_top1_retained = fixed_mask.gather(
        -1, full_teacher_probabilities.argmax(dim=-1).unsqueeze(-1)
    ).squeeze(-1)
    return {
        "schemes": metrics,
        "proxies": proxies,
        "need_score": need_score,
        "full_teacher_entropy": full_entropy,
        "full_teacher_normalized_entropy": full_normalized_entropy,
        "teacher_mass_outside_fixed_3x3": 1.0
        - fixed["teacher_mass_covered"],
        "teacher_top1_outside_fixed_3x3": (~fixed_top1_retained).float(),
        "shape": {
            "batch": int(teacher.shape[0]),
            "channels": int(teacher.shape[1]),
            "frames": int(teacher.shape[2]),
            "height": int(teacher.shape[3]),
            "width": int(teacher.shape[4]),
        },
    }


def summarize_proxy(proxy, need_score, fractions=(0.25, 0.5)):
    proxy = proxy.flatten()
    need_score = need_score.flatten()
    total = need_score.sum().clamp_min(1.0e-12)
    summary = {"spearman": spearman(proxy, need_score)}
    for fraction in fractions:
        count = max(1, int(math.ceil(fraction * proxy.numel())))
        indices = proxy.topk(count).indices
        captured = need_score[indices].sum() / total
        summary[f"top_{int(round(fraction * 100))}_mass_fraction"] = float(captured)
        summary[f"top_{int(round(fraction * 100))}_lift"] = float(
            captured / fraction
        )
    return summary
