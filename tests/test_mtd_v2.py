import torch
import torch.nn.functional as F

from qdiff.mtd import (
    _mtd_v2_confidence,
    _mtd_v2_coarse_terms,
    _mtd_v2_global_relation,
    _mtd_v2_predict_x0,
    _mtd_v2_soft_flow,
    _mtd_v2_soft_cycle_gate,
    build_mtd_v2_schedule_context,
    motion_transport_distillation_v2,
    normalize_mtd_v2_config,
)


def v2_config(enabled=True, **overrides):
    mtd_v2 = {
        "coarse_size": 4,
        "fine_size": 8,
        "fine_radius": 2,
        "coarse_topk": 8,
        "temporal_offsets": [1, 2, 4],
        "temperature": 0.07,
        "confidence_min": 0.0,
        "global_relation_weight": 0.1,
    }
    mtd_v2.update(overrides)
    return {
        "noise_channels": 4,
        "method": {"frame_axis": "MTD_V2" if enabled else "BASELINE"},
        "mtd_v2": mtd_v2,
    }


def context_for(video, timesteps=None, a=None, b=None):
    batch = video.shape[0]
    if timesteps is None:
        timesteps = torch.zeros(batch, dtype=torch.long)
    if a is None:
        a = torch.tensor([1.0])
    if b is None:
        b = torch.tensor([1.0])
    return {
        "x_t": torch.zeros_like(video),
        "timesteps": timesteps,
        "schedule": {
            "sqrt_recip_alphas_cumprod": a,
            "sqrt_recipm1_alphas_cumprod": b,
        },
    }


def moving_video(direction=1):
    torch.manual_seed(3)
    base = torch.randn(1, 4, 8, 8)
    x0 = torch.stack(
        [torch.roll(base, shifts=direction * frame, dims=-1) for frame in range(5)],
        dim=2,
    )
    # With x_t=0 and a=b=1, epsilon=-x0 reconstructs exactly to x0.
    return -x0


def test_v2_config_is_independent_and_idempotent():
    assert not normalize_mtd_v2_config()["enabled"]
    normalized = normalize_mtd_v2_config(v2_config())
    assert normalized["enabled"]
    assert normalized["feature_source"] == "x0"
    assert normalize_mtd_v2_config(normalized) == normalized


def test_v2_epsilon_feature_source_is_validated():
    normalized = normalize_mtd_v2_config(v2_config(feature_source="epsilon"))
    assert normalized["feature_source"] == "epsilon"
    try:
        normalize_mtd_v2_config(v2_config(feature_source="latent"))
    except ValueError as error:
        assert "feature_source" in str(error)
    else:
        raise AssertionError("invalid MTD-v2 feature source was accepted")


def test_epsilon_transport_terms_do_not_depend_on_xt():
    torch.manual_seed(23)
    target = torch.randn(1, 4, 5, 8, 8)
    prediction = target + 0.1 * torch.randn_like(target)
    cfg = v2_config(feature_source="epsilon", global_relation_weight=0.0)
    first_context = context_for(target)
    second_context = context_for(target)
    second_context["x_t"] = torch.randn_like(target) * 7.0
    _, first = motion_transport_distillation_v2(
        prediction, target, first_context, cfg, return_components=True
    )
    _, second = motion_transport_distillation_v2(
        prediction, target, second_context, cfg, return_components=True
    )
    assert torch.equal(first["correspondence"], second["correspondence"])
    assert torch.equal(first["flow"], second["flow"])
    assert first["global"].item() == 0.0
    assert second["global"].item() == 0.0


def test_x0_matches_scheduler_epsilon_formula_and_ignores_variance_channels():
    cfg = normalize_mtd_v2_config(v2_config())
    x_t = torch.full((2, 8, 5, 8, 8), 3.0)
    pred = torch.full_like(x_t, 2.0)
    target = torch.full_like(x_t, 5.0)
    context = {
        "x_t": x_t,
        "timesteps": torch.tensor([0, 1]),
        "schedule": {
            "sqrt_recip_alphas_cumprod": torch.tensor([2.0, 4.0]),
            "sqrt_recipm1_alphas_cumprod": torch.tensor([0.5, 1.5]),
        },
    }
    pred_x0, target_x0, sample_weights, _ = _mtd_v2_predict_x0(
        pred, target, context, cfg
    )
    assert torch.allclose(pred_x0[0], torch.full_like(pred_x0[0], 5.0))
    assert torch.allclose(pred_x0[1], torch.full_like(pred_x0[1], 9.0))
    assert torch.allclose(target_x0[0], torch.full_like(target_x0[0], 3.5))
    assert torch.allclose(target_x0[1], torch.full_like(target_x0[1], 4.5))
    assert torch.allclose(sample_weights, torch.tensor([1.0 / 3.0, 1.0 / 15.0]))
    assert pred_x0.shape[1] == 4


def test_x0_uses_the_scheduler_timestep_map_for_respaced_iddpm():
    cfg = normalize_mtd_v2_config(v2_config())
    x_t = torch.full((2, 4, 5, 8, 8), 3.0)
    pred = torch.full_like(x_t, 2.0)
    target = torch.full_like(x_t, 5.0)
    context = {
        "x_t": x_t,
        # These are original IDDPM IDs supplied to the denoiser, not 0/1.
        "timesteps": torch.tensor([999, 979]),
        "schedule": {
            "sqrt_recip_alphas_cumprod": torch.tensor([2.0, 4.0]),
            "sqrt_recipm1_alphas_cumprod": torch.tensor([0.5, 1.5]),
            "timestep_map": torch.tensor([999, 979]),
        },
    }
    pred_x0, target_x0, _, _ = _mtd_v2_predict_x0(pred, target, context, cfg)
    assert torch.allclose(pred_x0[0], torch.full_like(pred_x0[0], 5.0))
    assert torch.allclose(pred_x0[1], torch.full_like(pred_x0[1], 9.0))
    assert torch.allclose(target_x0[0], torch.full_like(target_x0[0], 3.5))
    assert torch.allclose(target_x0[1], torch.full_like(target_x0[1], 4.5))


def test_v2_identical_prediction_is_zero_and_bfloat16_gradient_is_finite():
    target = moving_video()
    prediction = target.to(torch.bfloat16).clone().requires_grad_(True)
    loss, components, diagnostics = motion_transport_distillation_v2(
        prediction,
        target.to(torch.bfloat16),
        context_for(prediction),
        v2_config(),
        return_components=True,
        return_diagnostics=True,
    )
    assert loss.dtype == torch.float32
    assert loss.abs().item() < 1.0e-5
    assert all(value.abs().item() < 1.0e-5 for value in components.values())
    assert all(f"offset_{offset}_coarse_entropy_mean" in diagnostics for offset in (1, 2, 4))
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad.float()).all()


def test_coarsened_topk_correspondence_is_nonnegative():
    """The selected-support, support-mass, and outside bins form a KL."""
    torch.manual_seed(11)
    cfg = normalize_mtd_v2_config(v2_config(coarse_topk=3))
    target_tokens = F.normalize(torch.randn(2, 5, 16, 7), dim=-1)
    prediction_tokens = F.normalize(torch.randn(2, 5, 16, 7), dim=-1)
    correspondence, _, _, _ = _mtd_v2_coarse_terms(
        prediction_tokens, target_tokens, 1, cfg, torch.ones(2)
    )
    assert correspondence.item() >= -1.0e-6


def test_soft_flow_recovers_a_known_large_displacement_per_frame():
    query = torch.tensor([[-1.0, -1.0], [0.0, 0.0]])
    candidates = torch.tensor(
        [[[[1.0, -1.0], [-1.0, -1.0]], [[1.0, 0.0], [0.0, 0.0]]]]
    ).unsqueeze(1)
    probabilities = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    flow = _mtd_v2_soft_flow(probabilities, candidates, query, offset=2)
    assert torch.allclose(
        flow,
        torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]]),
    )


def test_soft_cycle_gate_keeps_reversible_and_rejects_uniform_matches():
    cfg = normalize_mtd_v2_config(
        v2_config(cycle_enabled=True, cycle_min=0.05, cycle_power=1.0)
    )
    identity = torch.eye(4).view(1, 1, 4, 4)
    gate, score = _mtd_v2_soft_cycle_gate(identity, identity, cfg)
    assert torch.allclose(score, torch.ones_like(score))
    assert torch.allclose(gate, torch.ones_like(gate))

    uniform = torch.full((1, 1, 4, 4), 0.25)
    gate, score = _mtd_v2_soft_cycle_gate(uniform, uniform, cfg)
    assert torch.allclose(score, torch.full_like(score, 0.25))
    expected = torch.full_like(gate, (0.25 - 0.05) / 0.95)
    assert torch.allclose(gate, expected)


def test_soft_cycle_gate_can_mask_nonreturning_queries():
    cfg = normalize_mtd_v2_config(
        v2_config(cycle_enabled=True, cycle_min=0.05)
    )
    forward = torch.eye(4).view(1, 1, 4, 4)
    backward = torch.zeros_like(forward)
    backward[..., 0] = 1.0
    gate, score = _mtd_v2_soft_cycle_gate(forward, backward, cfg)
    assert score[0, 0, 0] == 1.0
    assert torch.count_nonzero(gate[0, 0, 1:]) == 0


def test_reverse_direction_and_static_motion_increase_transport_loss():
    target = moving_video(direction=1)
    context = context_for(target)
    same = motion_transport_distillation_v2(target, target, context, v2_config())
    reverse = motion_transport_distillation_v2(
        moving_video(direction=-1), target, context, v2_config()
    )
    static = moving_video(direction=0)
    static_loss = motion_transport_distillation_v2(static, target, context, v2_config())
    assert reverse > same + 1.0e-3
    assert static_loss > same + 1.0e-3


def test_entropy_mask_removes_uniform_teacher_and_weights_sharp_teacher():
    cfg = normalize_mtd_v2_config(v2_config(confidence_min=0.15))
    uniform = torch.full((1, 1, 1, 4), 0.25)
    sharp = torch.tensor([[[[0.97, 0.01, 0.01, 0.01]]]])
    valid = torch.ones_like(uniform, dtype=torch.bool)
    uniform_weight, _, uniform_confidence = _mtd_v2_confidence(uniform, valid, cfg)
    sharp_weight, _, sharp_confidence = _mtd_v2_confidence(sharp, valid, cfg)
    assert uniform_weight.item() == 0.0
    assert uniform_confidence.item() == 0.0
    assert sharp_weight.item() > 0.0
    assert sharp_confidence.item() > cfg["confidence_min"]


def test_v2_requires_complete_context_and_enough_frames():
    video = torch.randn(1, 4, 5, 8, 8)
    try:
        motion_transport_distillation_v2(video, video, None, v2_config())
    except ValueError as error:
        assert "requires x_t" in str(error)
    else:
        raise AssertionError("missing MTD-v2 context was silently accepted")
    try:
        motion_transport_distillation_v2(video[:, :, :4], video[:, :, :4], context_for(video), v2_config())
    except ValueError as error:
        assert "more than four" in str(error)
    else:
        raise AssertionError("short MTD-v2 video was silently accepted")


def test_global_relation_excludes_diagonal_and_is_frame_normalized():
    torch.manual_seed(7)
    target = torch.randn(1, 4, 5, 6, 6)
    prediction = target + 0.1 * torch.randn_like(target)
    mtd = normalize_mtd_v2_config(v2_config())
    actual = _mtd_v2_global_relation(prediction, target, torch.ones(1), mtd)

    pred_summary = F.normalize(prediction.mean(dim=(-1, -2)).transpose(1, 2), dim=-1)
    target_summary = F.normalize(target.mean(dim=(-1, -2)).transpose(1, 2), dim=-1)
    pred_relation = pred_summary @ pred_summary.transpose(1, 2)
    target_relation = target_summary @ target_summary.transpose(1, 2)
    diagonal = torch.eye(5, dtype=torch.bool).unsqueeze(0)
    floor = torch.finfo(prediction.dtype).min
    expected = (
        F.softmax(target_relation.masked_fill(diagonal, floor), dim=-1)
        * (
            F.softmax(target_relation.masked_fill(diagonal, floor), dim=-1).clamp_min(mtd["eps"]).log()
            - F.log_softmax(pred_relation.masked_fill(diagonal, floor), dim=-1)
        )
    ).sum(dim=-1).mean()
    assert torch.allclose(actual, expected, atol=1.0e-7, rtol=1.0e-6)

    duplicated = _mtd_v2_global_relation(
        prediction.repeat_interleave(2, dim=2),
        target.repeat_interleave(2, dim=2),
        torch.ones(1),
        mtd,
    )
    ratio = (duplicated / actual).item()
    assert 0.4 < ratio < 1.3


def test_schedule_context_hash_is_deterministic():
    class Scheduler:
        sqrt_recip_alphas_cumprod = [1.0, 2.0]
        sqrt_recipm1_alphas_cumprod = [0.0, 1.0]
        timestep_map = [999, 979]

    first = build_mtd_v2_schedule_context(Scheduler())
    second = build_mtd_v2_schedule_context(Scheduler())
    assert first["signature"] == second["signature"]
    assert first["sqrt_recip_alphas_cumprod"].dtype == torch.float32
    assert torch.equal(first["timestep_map"], torch.tensor([999, 979]))
