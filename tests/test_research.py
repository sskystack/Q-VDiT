import copy
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from qdiff.research import (
    TrackAwareResidual,
    _compact_transport_features,
    collect_rank_budget_loss,
    motion_transport_distillation,
    normalize_research_config,
    sample_trajectory_pair_indices,
    trajectory_bin,
    trajectory_consistency_loss,
)


def config(token="TARQ", frame="BASELINE", diffusion="NONE", groups=4):
    return {
        "method": {"token_axis": token, "frame_axis": frame, "diffusion_axis": diffusion},
        "tarq": {
            "num_groups": groups,
            "rank_per_group": 1,
            "transport_size": 2,
            "descriptor_dim": 2,
            "temperature": 0.2,
            "rank_budget": 1.0,
        },
        "mtd": {"transport_size": 4, "temperature": 0.2},
        "taq": {"num_bins": 8},
    }


def test_four_configs_have_the_expected_single_variable_progression():
    root = Path(__file__).parents[1] / "t2v/configs/quant/opensora"
    expected = {
        "w4a6_baseline.yaml": ("TQE", "BASELINE", "NONE"),
        "w4a6_tarq.yaml": ("TARQ", "BASELINE", "NONE"),
        "w4a6_tarq_mtd.yaml": ("TARQ", "MTD", "NONE"),
        "w4a6_tarq_mtd_taq.yaml": ("TARQ", "MTD", "TAQ"),
    }
    for filename, methods in expected.items():
        data = yaml.safe_load((root / filename).read_text())
        assert data["calib_data"]["n_steps"] == 50
        assert data["calib_data"]["batch_size"] == 4
        assert data["calib_data"]["keep_cache_on_cpu"] is True
        assert data["calib_data"]["pin_memory"] is True
        assert data["calib_data"]["async_prefetch"] is True
        assert tuple(data["method"].values()) == methods


def test_smoke_configs_cover_each_new_module_stage():
    root = Path(__file__).parent / "configs"
    expected = {
        "w4a6_tarq_smoke.yaml": ("TARQ", "BASELINE", "NONE"),
        "w4a6_tarq_mtd_smoke.yaml": ("TARQ", "MTD", "NONE"),
        "w4a6_full_smoke.yaml": ("TARQ", "MTD", "TAQ"),
    }
    for filename, methods in expected.items():
        data = yaml.safe_load((root / filename).read_text())
        assert tuple(data["method"].values()) == methods
        assert data["calib_data"]["batch_size"] == 4
        assert data["quant"]["weight"]["optimization"]["iters"] >= 1


def test_formal_tarq_stability_config_only_reduces_iterations():
    root = Path(__file__).parents[1]
    formal = yaml.safe_load(
        (root / "t2v/configs/quant/opensora/w4a6_tarq.yaml").read_text()
    )
    stability = yaml.safe_load(
        (root / "tests/configs/w4a6_tarq_formal_stability.yaml").read_text()
    )
    formal_iters = formal["quant"]["weight"]["optimization"].pop("iters")
    stability_iters = stability["quant"]["weight"]["optimization"].pop("iters")
    assert formal_iters == 10000
    assert stability_iters == 3
    assert stability == formal


def test_research_config_normalization_is_idempotent():
    normalized = normalize_research_config(config(frame="MTD", diffusion="TAQ"))
    assert normalize_research_config(normalized) == normalized


def test_trajectory_bins_match_normalized_50_and_100_step_progress():
    for progress in (0.0, 0.1, 0.5, 0.9, 1.0):
        step_50 = min(49, int(progress * 50))
        step_100 = min(99, int(progress * 100))
        assert trajectory_bin(step_50, 50, 8) == trajectory_bin(step_100, 100, 8)


def test_trajectory_pair_sampler_returns_adjacent_steps_in_the_same_bin():
    indices = sample_trajectory_pair_indices(1000, 50, 20, 32, 4, torch.device("cpu"), 8)
    assert indices.shape == (32, 4)
    for row in indices:
        for first, second in row.reshape(-1, 2):
            assert int(second - first) == 20
            first_step = int(first) // 20
            second_step = int(second) // 20
            assert trajectory_bin(first_step, 50, 8) == trajectory_bin(second_step, 50, 8)


def test_single_group_tarq_reduces_to_one_low_rank_residual():
    module = TrackAwareResidual(3, 2, config(groups=1))
    with torch.no_grad():
        module.down.copy_(torch.tensor([[[1.0, -2.0, 0.5]]]))
        module.up.copy_(torch.tensor([[[2.0], [-1.0]]]))
    inputs = torch.randn(2, 4, 3)  # B*T=2, S=4, C=3 with B=1,T=2
    actual = module(inputs, batch=1, frames=2, spatial_tokens=4, layout="spatial")
    expected = F.linear(F.linear(inputs.reshape(1, 2, 4, 3), module.down[0]), module.up[0]).reshape(2, 4, 2)
    assert torch.allclose(actual, expected, atol=1.0e-6)
    assert module.last_budget_loss.item() == 0.0


def test_tarq_rank_budget_penalizes_multiple_effective_groups_and_backpropagates():
    module = TrackAwareResidual(3, 2, config(diffusion="NONE", groups=4))
    with torch.no_grad():
        module.up.fill_(0.1)
    output = module(
        torch.randn(2, 4, 3), batch=1, frames=2, spatial_tokens=4, layout="spatial"
    )
    rank_loss = collect_rank_budget_loss(module)
    assert rank_loss.item() > 0.0
    rank_loss.backward()
    assert module.gate.weight.grad is not None
    assert torch.isfinite(module.gate.weight.grad).all()


def test_tarq_rank_budget_keeps_gate_gradients_under_checkpoint_no_grad_forward():
    module = TrackAwareResidual(3, 2, config(diffusion="NONE", groups=4))
    with torch.no_grad():
        module(torch.randn(2, 4, 3), batch=1, frames=2, spatial_tokens=4, layout="spatial")
    rank_loss = collect_rank_budget_loss(module)
    assert rank_loss.requires_grad
    rank_loss.backward()
    assert module.gate.weight.grad is not None
    assert torch.isfinite(module.gate.weight.grad).all()


def test_effective_group_count_matches_one_hot_and_uniform_allocations():
    one_hot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    uniform = torch.full((1, 4), 0.25)
    assert TrackAwareResidual.effective_group_count(one_hot).item() == 1.0
    assert TrackAwareResidual.effective_group_count(uniform).item() == 4.0


def test_rank_loss_bookkeeping_preserves_original_tarq_forward_and_reconstruction_gradients():
    torch.manual_seed(7)
    updated = TrackAwareResidual(3, 2, config(diffusion="NONE", groups=4))
    reference = copy.deepcopy(updated)

    def original_allocation(self, motion):
        temperature, rank_budget = self._temperature_and_budget()
        soft_gates = F.softmax(self.gate(motion) / temperature, dim=-1)
        gates, _ = self._allocate_rank_groups(soft_gates, rank_budget)
        return gates, self.down.new_tensor(0.0)

    reference._allocation_budget_loss = types.MethodType(original_allocation, reference)
    source = torch.randn(2, 4, 3)
    probe = torch.randn(2, 4, 2)
    updated_input = source.clone().requires_grad_()
    reference_input = source.clone().requires_grad_()
    updated_output = updated(updated_input, 1, 2, 4, "spatial")
    reference_output = reference(reference_input, 1, 2, 4, "spatial")
    (updated_output * probe).sum().backward()
    (reference_output * probe).sum().backward()

    assert torch.equal(updated_output, reference_output)
    assert torch.equal(updated_input.grad, reference_input.grad)
    for updated_parameter, reference_parameter in zip(updated.parameters(), reference.parameters()):
        if updated_parameter.grad is None or reference_parameter.grad is None:
            assert updated_parameter.grad is reference_parameter.grad
        else:
            assert torch.equal(updated_parameter.grad, reference_parameter.grad)


def test_taq_rank_budget_changes_the_number_of_active_groups():
    module = TrackAwareResidual(3, 2, config(diffusion="TAQ", groups=4))
    gates = torch.tensor([[[[0.4, 0.3, 0.2, 0.1]]]])
    with torch.no_grad():
        module.taq_rank_logit.fill_(-8.0)
        _, low_budget = module._temperature_and_budget()
        _, low_mask = module._allocate_rank_groups(gates, low_budget)
        module.taq_rank_logit.fill_(8.0)
        _, high_budget = module._temperature_and_budget()
        _, high_mask = module._allocate_rank_groups(gates, high_budget)
    assert low_budget.item() < 1.01
    assert high_budget.item() > 3.99
    assert low_mask.sum().item() < high_mask.sum().item()


def test_taq_rank_budget_receives_reconstruction_gradients():
    module = TrackAwareResidual(3, 2, config(diffusion="TAQ", groups=4))
    with torch.no_grad():
        module.up.fill_(0.1)
        module.gate.weight.normal_(0.0, 0.1)
    module.set_trajectory_position(25, 50)
    output = module(torch.randn(2, 4, 3), batch=1, frames=2, spatial_tokens=4, layout="spatial")
    (output.square().mean() + module.budget_regularization()).backward()
    assert module.taq_rank_logit.grad is not None
    assert module.taq_rank_logit.grad.abs().sum().item() > 0.0


def test_eval_rank_allocation_is_discrete():
    module = TrackAwareResidual(3, 2, config(diffusion="TAQ", groups=4)).eval()
    gates = torch.tensor([[[[0.4, 0.3, 0.2, 0.1]]]])
    with torch.no_grad():
        module.taq_rank_logit.fill_(-8.0)
        _, budget = module._temperature_and_budget()
        allocated, active = module._allocate_rank_groups(gates, budget)
    assert active.sum().item() == 1.0
    assert torch.count_nonzero(allocated).item() == 1


def test_tarq_eval_without_taq_accepts_scalar_rank_budget():
    module = TrackAwareResidual(3, 2, config(diffusion="NONE", groups=4)).eval()
    gates = torch.tensor([[[[0.4, 0.3, 0.2, 0.1]]]])
    with torch.no_grad():
        _, budget = module._temperature_and_budget()
        allocated, active = module._allocate_rank_groups(gates, budget)
    assert active.sum().item() == 1.0
    assert torch.count_nonzero(allocated).item() == 1


def test_mtd_is_zero_for_identical_features_and_positive_for_motion_error():
    target = torch.randn(2, 4, 3, 8, 8)
    identical = motion_transport_distillation(target, target, config(frame="MTD"))
    shifted = motion_transport_distillation(torch.roll(target, shifts=1, dims=-1), target, config(frame="MTD"))
    assert identical.abs().item() < 1.0e-5
    assert shifted.item() > identical.item()


def test_transport_magnitude_matches_sqrt_and_has_finite_zero_gradient():
    displacement = torch.randn(32, 2, requires_grad=True)
    stable = torch.linalg.vector_norm(displacement, dim=-1)
    reference = displacement.square().sum(-1).sqrt()
    assert torch.allclose(stable, reference, atol=1.0e-7, rtol=1.0e-6)

    zero = torch.zeros(8, 2, requires_grad=True)
    torch.linalg.vector_norm(zero, dim=-1).sum().backward()
    assert torch.isfinite(zero.grad).all()
    assert torch.count_nonzero(zero.grad).item() == 0


def test_zero_motion_transport_has_finite_end_to_end_gradient():
    video = torch.zeros(1, 2, 8, 8, 8, requires_grad=True)
    features = _compact_transport_features(
        video, transport_size=8, descriptor_dim=8, temperature=0.07
    )
    features.sum().backward()
    assert torch.isfinite(features).all()
    assert torch.isfinite(video.grad).all()


def test_trajectory_loss_uses_adjacent_pairs():
    target = torch.randn(4, 6, 2, 2, 2)
    assert trajectory_consistency_loss(target, target, config(diffusion="TAQ")).item() == 0.0
    prediction = target.clone()
    prediction[1, :3] += 1.0
    assert trajectory_consistency_loss(prediction, target, config(diffusion="TAQ")).item() > 0.0


def test_sequence_layout_matches_spatial_layout_correction():
    torch.manual_seed(3)
    module = TrackAwareResidual(6, 5, config(groups=4))
    with torch.no_grad():
        module.up.normal_(0.0, 0.1)
        module.gate.weight.normal_(0.0, 0.1)
    x = torch.randn(1, 2 * 4, 6)  # B=1, T=2, S=4 as one frame-major sequence
    seq = module(x, batch=1, frames=2, spatial_tokens=4, layout="sequence")
    spa = module(x.reshape(2, 4, 6), batch=1, frames=2, spatial_tokens=4, layout="spatial")
    assert seq.shape == (1, 8, 5)
    assert torch.allclose(seq.reshape(2, 4, 5), spa, atol=1.0e-6)


def test_block_motion_context_computes_transport_once_and_is_shape_guarded():
    from qdiff import research

    producer = TrackAwareResidual(6, 5, config(groups=4))
    consumer = TrackAwareResidual(3, 2, config(groups=4))
    context = research.BlockMotionContext()
    producer.motion_context = context
    consumer.motion_context = context

    calls = []
    original = research._compact_transport_features

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    research._compact_transport_features = counting
    try:
        producer(torch.randn(2, 4, 6), batch=1, frames=2, spatial_tokens=4, layout="spatial")
        consumer(torch.randn(4, 2, 3), batch=1, frames=2, spatial_tokens=4, layout="temporal")
        assert len(calls) == 1  # consumer reused the cached motion features
        context.clear()
        consumer(torch.randn(4, 2, 3), batch=1, frames=2, spatial_tokens=4, layout="temporal")
        assert len(calls) == 2  # cleared cache forces recomputation
        # a mismatched grid must never reuse the cache
        producer(torch.randn(2, 16, 6), batch=1, frames=2, spatial_tokens=16, layout="spatial")
        assert len(calls) == 3
    finally:
        research._compact_transport_features = original


def test_tarq_apply_to_scope_is_validated_and_defaults_to_attention():
    normalized = normalize_research_config({"method": {"token_axis": "TARQ"}})
    assert normalized["tarq"]["apply_to"] == ("spatial_attn", "temporal_attn")
    full = normalize_research_config(
        {"method": {"token_axis": "TARQ"},
         "tarq": {"apply_to": ["spatial_attn", "temporal_attn", "cross_attn", "ffn"]}}
    )
    assert set(full["tarq"]["apply_to"]) == {"spatial_attn", "temporal_attn", "cross_attn", "ffn"}
    try:
        normalize_research_config({"tarq": {"apply_to": ["ffn", "typo_axis"]}})
    except ValueError as error:
        assert "typo_axis" in str(error)
    else:
        raise AssertionError("invalid tarq scope must raise")
