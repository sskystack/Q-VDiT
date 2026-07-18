from collections import OrderedDict

import torch

from qdiff.reconstruction_checkpoint import (
    atomic_torch_save,
    inference_quant_params_state,
    optimizer_parameter_names,
    restore_trainable_parameter_state,
    trainable_parameter_state,
)


def _step(model, optimizer, scheduler, source, target):
    optimizer.zero_grad()
    loss = (model(source) - target).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()


def test_resume_state_matches_uninterrupted_adamw_and_cosine_schedule():
    torch.manual_seed(11)
    initial = torch.nn.Linear(3, 2)
    uninterrupted = torch.nn.Linear(3, 2)
    interrupted = torch.nn.Linear(3, 2)
    uninterrupted.load_state_dict(initial.state_dict())
    interrupted.load_state_dict(initial.state_dict())
    batches = [(torch.randn(4, 3), torch.randn(4, 2)) for _ in range(4)]

    full_optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=1.0e-3)
    full_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(full_optimizer, T_max=4)
    for batch in batches:
        _step(uninterrupted, full_optimizer, full_scheduler, *batch)

    first_optimizer = torch.optim.AdamW(interrupted.parameters(), lr=1.0e-3)
    first_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(first_optimizer, T_max=4)
    for batch in batches[:2]:
        _step(interrupted, first_optimizer, first_scheduler, *batch)
    checkpoint = {
        "parameters": trainable_parameter_state(interrupted),
        "optimizer": first_optimizer.state_dict(),
        "scheduler": first_scheduler.state_dict(),
        "optimizer_parameter_names": optimizer_parameter_names(interrupted, first_optimizer),
    }

    resumed = torch.nn.Linear(3, 2)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1.0e-3)
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=4)
    assert optimizer_parameter_names(resumed, resumed_optimizer) == checkpoint["optimizer_parameter_names"]
    restore_trainable_parameter_state(resumed, checkpoint["parameters"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_scheduler.load_state_dict(checkpoint["scheduler"])
    for batch in batches[2:]:
        _step(resumed, resumed_optimizer, resumed_scheduler, *batch)

    for expected, actual in zip(uninterrupted.parameters(), resumed.parameters()):
        assert torch.equal(expected, actual)
    assert full_scheduler.state_dict() == resumed_scheduler.state_dict()


def test_inference_checkpoint_moves_live_quantizer_parameters_into_buffers():
    live = {
        "layer.weight_quantizer": [
            OrderedDict([("zero_point", torch.tensor([2.0]))]),
            OrderedDict([("delta", torch.tensor([0.25]))]),
        ],
        "layer.loraA.weight": torch.tensor([3.0]),
        "__research_state__": {"layer.tarq.up": torch.tensor([4.0])},
    }
    inference = inference_quant_params_state(live)
    buffers, parameters = inference["layer.weight_quantizer"]
    assert parameters == OrderedDict()
    assert torch.equal(buffers["zero_point"], torch.tensor([2.0]))
    assert torch.equal(buffers["delta"], torch.tensor([0.25]))
    assert inference["layer.loraA.weight"].device.type == "cpu"
    assert inference["__research_state__"]["layer.tarq.up"].device.type == "cpu"


def test_atomic_checkpoint_replaces_the_latest_state(tmp_path):
    path = tmp_path / "reconstruction_state_latest.pth"
    atomic_torch_save({"iteration": 1}, str(path))
    atomic_torch_save({"iteration": 2}, str(path))
    assert torch.load(path, map_location="cpu")["iteration"] == 2
    assert not (tmp_path / "reconstruction_state_latest.pth.tmp").exists()
