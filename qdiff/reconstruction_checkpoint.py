import os
from collections import OrderedDict

import torch


def atomic_torch_save(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = path + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def trainable_parameter_state(module):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_parameter_state(module, state):
    parameters = {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    missing = sorted(set(parameters) - set(state))
    unexpected = sorted(set(state) - set(parameters))
    if missing or unexpected:
        raise KeyError(
            "Reconstruction checkpoint trainable parameters do not match: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for name, value in state.items():
        parameters[name].data.copy_(value.to(parameters[name].device, parameters[name].dtype))


def nested_tensors_to_cpu(value):
    if isinstance(value, dict):
        return {key: nested_tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [nested_tensors_to_cpu(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def nested_tensors_to_device(value, device):
    if isinstance(value, dict):
        return {key: nested_tensors_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [nested_tensors_to_device(item, device) for item in value]
    if torch.is_tensor(value):
        return value.to(device)
    return value


def optimizer_parameter_names(module, optimizer):
    names_by_id = {id(parameter): name for name, parameter in module.named_parameters()}
    return [
        [names_by_id[id(parameter)] for parameter in group['params']]
        for group in optimizer.param_groups
    ]


def inference_quant_params_state(live_state, dtype=torch.float32):
    """Convert live reconstruction parameters into inference buffer format."""
    inference_state = {}
    for name, value in live_state.items():
        if name == "__research_state__":
            inference_state[name] = {
                key: tensor.detach().to(device="cpu", dtype=dtype).clone()
                for key, tensor in value.items()
            }
        elif isinstance(value, list) and len(value) == 2:
            buffers, parameters = value
            merged = OrderedDict()
            for key, tensor in buffers.items():
                merged[key] = (
                    tensor.detach().to(device="cpu", dtype=dtype).clone()
                    if tensor is not None else None
                )
            for key, tensor in parameters.items():
                merged[key] = (
                    tensor.detach().to(device="cpu", dtype=dtype).clone()
                    if tensor is not None else None
                )
            inference_state[name] = [merged, OrderedDict()]
        elif torch.is_tensor(value):
            inference_state[name] = value.detach().to(device="cpu", dtype=dtype).clone()
        else:
            inference_state[name] = value
    return inference_state
