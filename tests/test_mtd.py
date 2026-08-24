import torch
import torch.nn.functional as F

from qdiff.mtd import (
    _importance_weights,
    _key_region_mask,
    _local_transport_distribution,
    _teacher_matching_confidence,
    motion_transport_distillation,
    normalize_mtd_config,
)


def config(enabled=True, noise_channels=4):
    return {
        "noise_channels": noise_channels,
        "method": {"frame_axis": "MTD" if enabled else "BASELINE"},
        "mtd": {
            "transport_size": 4,
            "temperature": 0.2,
            "local_transport_weight": 1.0,
            "motion_residual_weight": 1.0,
            "global_relation_weight": 0.1,
        },
    }


def test_config_is_disabled_by_default():
    assert not normalize_mtd_config()["enabled"]


def test_config_normalization_is_idempotent():
    normalized = normalize_mtd_config(config())
    assert normalize_mtd_config(normalized) == normalized


def test_mtd_is_zero_when_disabled_or_single_frame():
    video = torch.randn(2, 4, 3, 8, 8)
    assert motion_transport_distillation(video, video, config(False)).item() == 0.0
    assert motion_transport_distillation(video[:, :, :1], video[:, :, :1], config()).item() == 0.0


def test_mtd_is_zero_for_identical_features_and_positive_for_motion_error():
    target = torch.randn(2, 4, 3, 8, 8)
    identical = motion_transport_distillation(target, target, config())
    shifted = motion_transport_distillation(
        torch.roll(target, shifts=1, dims=-1), target, config()
    )
    assert identical.abs().item() < 1.0e-5
    assert shifted.item() > identical.item()


def test_local_kl_is_averaged_over_spatial_tokens():
    torch.manual_seed(11)
    target = torch.randn(2, 4, 3, 8, 8)
    prediction = target + 0.05 * torch.randn_like(target)
    cfg = config()
    cfg["mtd"].update(
        motion_residual_weight=0.0,
        global_relation_weight=0.0,
    )
    pred_probs, _, _ = _local_transport_distribution(prediction, 4, 0.2)
    with torch.no_grad():
        target_probs, _, _ = _local_transport_distribution(target, 4, 0.2)
    expected = F.kl_div(
        pred_probs.clamp_min(1.0e-8).log(), target_probs, reduction="none"
    ).sum(dim=-1).mean()
    actual = motion_transport_distillation(prediction, target, cfg)
    assert torch.allclose(actual, expected, atol=1.0e-7, rtol=1.0e-6)


def test_padded_neighbours_are_masked_at_frame_boundaries():
    features = torch.zeros(1, 4, 2, 4, 4)
    probabilities, _, _ = _local_transport_distribution(features, 4, 0.2)
    assert torch.count_nonzero(probabilities[0, 0]).item() == 4
    assert torch.count_nonzero(probabilities[0, 1]).item() == 6
    assert torch.count_nonzero(probabilities[0, 5]).item() == 9
    assert torch.allclose(
        probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0])
    )


def test_bfloat16_input_produces_fp32_loss_and_finite_gradient():
    target = torch.randn(1, 8, 3, 8, 8, dtype=torch.bfloat16)
    prediction = target.clone().requires_grad_(True)
    loss = motion_transport_distillation(prediction, target, config())
    assert loss.dtype == torch.float32
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad.float()).all()


def test_only_noise_channels_are_supervised():
    target = torch.randn(1, 8, 3, 8, 8)
    prediction = target.clone()
    prediction[:, 4:] += 100.0
    assert motion_transport_distillation(
        prediction, target, config(noise_channels=4)
    ).abs().item() < 1.0e-5


def test_importance_weights_are_detached_and_mean_one():
    torch.manual_seed(13)
    prediction = torch.randn(2, 7, 4, requires_grad=True)
    target = torch.randn_like(prediction)
    weights = _importance_weights(prediction, target, 1.0e-8)
    assert not weights.requires_grad
    assert torch.allclose(
        weights.mean(dim=-1), torch.ones(2), atol=1.0e-6, rtol=1.0e-6
    )


def test_weighted_fixed_mtd_has_finite_gradients_and_reports_weights():
    torch.manual_seed(17)
    target = torch.randn(1, 4, 3, 8, 8)
    prediction = (target + 0.1 * torch.randn_like(target)).requires_grad_(True)
    cfg = config()
    cfg["mtd"]["spatial_weighting"] = "fp_quant_error"
    loss, components, diagnostics = motion_transport_distillation(
        prediction,
        target,
        cfg,
        return_components=True,
        return_diagnostics=True,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    assert torch.isfinite(prediction.grad).all()
    assert torch.allclose(
        diagnostics["importance_weight_mean"],
        torch.tensor(1.0),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def semantic_config():
    cfg = config()
    cfg["mtd"].update(
        semantic_weighting=True,
        motion_residual_weight=1.0,
        global_relation_weight=0.1,
    )
    return cfg


def test_all_one_semantic_weights_reproduce_fixed_mtd_terms():
    torch.manual_seed(18)
    target = torch.randn(2, 4, 3, 8, 8)
    prediction = target + 0.1 * torch.randn_like(target)
    _, baseline = motion_transport_distillation(
        prediction, target, config(), return_components=True
    )
    weights = torch.ones(2, 2, 16)
    _, semantic = motion_transport_distillation(
        prediction,
        target,
        semantic_config(),
        return_components=True,
        importance_weights=weights,
    )
    for name in ("local", "motion", "global", "fine"):
        assert torch.allclose(semantic[name], baseline[name], atol=1.0e-7, rtol=1.0e-6)


def test_semantic_weight_scale_cancels_and_only_changes_local_term():
    torch.manual_seed(20)
    target = torch.randn(1, 4, 3, 8, 8)
    prediction = (target + 0.15 * torch.randn_like(target)).requires_grad_(True)
    weights = torch.linspace(0.5, 1.5, 32).reshape(1, 2, 16)
    _, first = motion_transport_distillation(
        prediction,
        target,
        semantic_config(),
        return_components=True,
        importance_weights=weights,
    )
    _, scaled = motion_transport_distillation(
        prediction,
        target,
        semantic_config(),
        return_components=True,
        importance_weights=weights * 7.0,
    )
    _, baseline = motion_transport_distillation(
        prediction, target, config(), return_components=True
    )
    assert torch.allclose(first["local"], scaled["local"], atol=1.0e-7, rtol=1.0e-6)
    for name in ("motion", "global", "fine"):
        assert torch.equal(first[name], baseline[name])


def test_semantic_weight_shape_is_checked():
    target = torch.randn(1, 4, 3, 8, 8)
    try:
        motion_transport_distillation(
            target,
            target,
            semantic_config(),
            importance_weights=torch.ones(1, 2, 15),
        )
    except ValueError as error:
        assert "must have shape" in str(error)
    else:
        raise AssertionError("invalid semantic importance shape was accepted")


def test_random_candidates_are_iteration_deterministic():
    torch.manual_seed(19)
    target = torch.randn(1, 4, 3, 8, 8)
    prediction = target + 0.2 * torch.randn_like(target)
    cfg = config()
    cfg["mtd"].update(
        candidate_mode="random_topk",
        search_radius=2,
        topk=9,
        random_seed=42,
        iteration=11,
    )
    first = motion_transport_distillation(prediction, target, cfg)
    second = motion_transport_distillation(prediction, target, cfg)
    assert torch.equal(first, second)
    cfg["mtd"]["iteration"] = 12
    third = motion_transport_distillation(prediction, target, cfg)
    assert not torch.equal(first, third)


def test_teacher_centered_candidates_retain_teacher_top1_and_backpropagate():
    torch.manual_seed(23)
    first = torch.randn(1, 4, 8, 8)
    second = torch.zeros_like(first)
    second[..., 2:] = first[..., :-2]
    target = torch.stack([first, second], dim=2)
    prediction = (target + 0.05 * torch.randn_like(target)).requires_grad_(True)
    cfg = config()
    cfg["mtd"].update(
        candidate_mode="teacher_centered_3x3",
        search_radius=2,
    )
    loss, _, diagnostics = motion_transport_distillation(
        prediction,
        target,
        cfg,
        return_components=True,
        return_diagnostics=True,
    )
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert diagnostics["teacher_top1_retained"].item() == 1.0
    assert diagnostics["candidate_count"].item() >= 4.0
    assert tuple(diagnostics["coarse_offset_histogram"].shape) == (5, 5)


def test_v1_rejects_combined_weighting_and_adaptive_candidates():
    cfg = config()
    cfg["mtd"].update(
        candidate_mode="teacher_centered_3x3",
        spatial_weighting="fp_quant_error",
    )
    try:
        normalize_mtd_config(cfg)
    except ValueError as error:
        assert "separate v1 ablation arms" in str(error)
    else:
        raise AssertionError("combined adaptive/weighted v1 arm was accepted")


def fine_config(selection="error"):
    cfg = config()
    cfg["mtd"].update(
        fine_enabled=True,
        fine_weight=0.1,
        fine_transport_size=8,
        fine_kernel_size=5,
        fine_selection=selection,
        key_region_ratio=0.25,
        key_region_block_size=2,
        fine_selection_seed=42,
    )
    return cfg


def test_key_region_masks_have_exact_block_coverage_and_are_detached():
    local = torch.arange(32, dtype=torch.float32).reshape(2, 16).requires_grad_(True)
    motion = torch.flip(local.detach(), dims=(-1,)).requires_grad_(True)
    cfg = normalize_mtd_config(fine_config("error"))
    mask = _key_region_mask(local, motion, 4, cfg)
    assert not mask.requires_grad
    assert torch.equal(mask.sum(dim=-1), torch.tensor([4, 4]))

    random_mask = _key_region_mask(
        local, motion, 4, normalize_mtd_config(fine_config("random"))
    )
    assert torch.equal(random_mask.sum(dim=-1), torch.tensor([4, 4]))
    assert torch.equal(
        random_mask,
        _key_region_mask(
            local, motion, 4, normalize_mtd_config(fine_config("random"))
        ),
    )

    all_mask = _key_region_mask(
        local, motion, 4, normalize_mtd_config(fine_config("all"))
    )
    assert all_mask.all()


def test_teacher_confidence_is_candidate_count_normalized_and_detached():
    uniform = torch.zeros(2, 9, requires_grad=True)
    valid = torch.zeros(2, 9, dtype=torch.bool)
    valid[0, :4] = True
    valid[1, :] = True
    probabilities = uniform.masked_fill(~valid, float("-inf")).softmax(dim=-1)
    confidence = _teacher_matching_confidence(probabilities, valid)
    assert not confidence.requires_grad
    assert torch.allclose(confidence, torch.zeros_like(confidence), atol=1.0e-6)


def test_entropy_confidence_changes_error_based_key_region_ranking():
    local = torch.ones(1, 16)
    motion = torch.ones(1, 16)
    local[0, 0] = 20.0
    motion[0, 0] = 20.0
    local[0, 1] = 10.0
    motion[0, 1] = 10.0

    teacher_probs = torch.full((1, 16, 9), 1.0 / 9.0)
    teacher_probs[0, 1] = 0.0
    teacher_probs[0, 1, 4] = 1.0
    teacher_valid = torch.ones_like(teacher_probs, dtype=torch.bool)
    cfg = fine_config("error")
    cfg["mtd"].update(
        key_region_confidence="entropy",
        key_region_confidence_power=1.0,
    )
    mask = _key_region_mask(
        local,
        motion,
        4,
        normalize_mtd_config(cfg),
        teacher_probs=teacher_probs,
        teacher_valid=teacher_valid,
    )
    assert not mask[0, 0]
    assert mask[0, 1]


def test_fine_refinement_preserves_coarse_terms_and_backpropagates():
    torch.manual_seed(29)
    target = torch.randn(1, 4, 3, 8, 8)
    baseline_prediction = (
        target + 0.1 * torch.randn_like(target)
    ).requires_grad_(True)
    fine_prediction = baseline_prediction.detach().clone().requires_grad_(True)
    baseline_loss, baseline_terms = motion_transport_distillation(
        baseline_prediction, target, config(), return_components=True
    )
    fine_loss, fine_terms, diagnostics = motion_transport_distillation(
        fine_prediction,
        target,
        fine_config("error"),
        return_components=True,
        return_diagnostics=True,
    )
    for name in ("local", "motion", "global"):
        assert torch.equal(baseline_terms[name], fine_terms[name])
    assert fine_terms["fine"] > 0
    assert torch.allclose(fine_loss, baseline_loss + fine_terms["fine"])
    assert diagnostics["fine_selected_fraction"].item() == 0.25
    fine_loss.backward()
    assert torch.isfinite(fine_prediction.grad).all()


def test_confidence_guided_fine_refinement_reports_teacher_reliability():
    torch.manual_seed(31)
    target = torch.randn(1, 4, 3, 8, 8)
    prediction = (target + 0.1 * torch.randn_like(target)).requires_grad_(True)
    cfg = fine_config("error")
    cfg["mtd"]["key_region_confidence"] = "entropy"
    loss, diagnostics = motion_transport_distillation(
        prediction,
        target,
        cfg,
        return_diagnostics=True,
    )
    loss.backward()
    for name in (
        "teacher_confidence_mean",
        "teacher_confidence_selected",
        "teacher_confidence_unselected",
    ):
        assert name in diagnostics
        assert 0.0 <= diagnostics[name].item() <= 1.0
    assert torch.isfinite(prediction.grad).all()


def test_fine_refinement_is_zero_for_identical_features():
    target = torch.randn(1, 4, 3, 8, 8)
    loss, terms = motion_transport_distillation(
        target, target, fine_config("all"), return_components=True
    )
    assert loss.abs().item() < 1.0e-5
    assert terms["fine"].abs().item() < 1.0e-5


def test_semantic_scores_select_native_fine_queries_without_weighting_coarse():
    torch.manual_seed(37)
    target = torch.randn(1, 4, 3, 8, 8)
    prediction = (target + 0.1 * torch.randn_like(target)).requires_grad_(True)
    cfg = fine_config("error")
    cfg["mtd"].update(
        fine_selection="semantic",
        semantic_importance=True,
        semantic_weighting=False,
        fine_motion_weight=0.0,
    )
    scores = torch.linspace(0.0, 1.0, 2 * 64).reshape(1, 2, 64)
    baseline_loss, baseline_terms = motion_transport_distillation(
        prediction, target, config(), return_components=True
    )
    loss, terms, diagnostics = motion_transport_distillation(
        prediction,
        target,
        cfg,
        return_components=True,
        return_diagnostics=True,
        fine_selection_scores=scores,
    )
    for name in ("local", "motion", "global"):
        assert torch.equal(terms[name], baseline_terms[name])
    assert terms["fine"] > 0
    assert torch.allclose(loss, baseline_loss + terms["fine"])
    assert diagnostics["fine_selected_fraction"].item() == 0.25
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_semantic_fine_score_shape_is_checked():
    target = torch.randn(1, 4, 3, 8, 8)
    cfg = fine_config("error")
    cfg["mtd"].update(
        fine_selection="semantic",
        semantic_importance=True,
        semantic_weighting=False,
    )
    try:
        motion_transport_distillation(
            target,
            target,
            cfg,
            fine_selection_scores=torch.ones(1, 2, 63),
        )
    except ValueError as error:
        assert "fine_selection_scores must have shape" in str(error)
    else:
        raise AssertionError("invalid semantic fine score shape was accepted")
