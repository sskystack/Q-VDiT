import json
import math
import os
import random

import numpy as np
import torch
# import linklink as link
import logging
from qdiff.quantizer.base_quantizer import lp_loss
from qdiff.models.quant_layer import QuantLayer
from qdiff.models.stdit_quant_layer import QuantTemporalAttnLinear
from qdiff.models.quant_model import QuantModel
from qdiff.models.quant_block import BaseQuantBlock
from qdiff.quantizer.base_quantizer import StraightThrough
# from qdiff.quantizer.base_quantizer import AdaRoundQuantizer
from qdiff.utils import save_grad_data, save_in_out_data, LossFunction
from qdiff.mtd import normalize_mtd_config
from qdiff.reconstruction_checkpoint import (
    atomic_torch_save,
    nested_tensors_to_cpu,
    nested_tensors_to_device,
    optimizer_parameter_names,
    restore_trainable_parameter_state,
    trainable_parameter_state,
)
from torch.cuda.amp import GradScaler, autocast
from opensora.acceleration.checkpoint import set_grad_checkpoint
from qdiff.memory_profile import mark_memory
from qdiff.numeric_monitor import NumericMonitor

logger = logging.getLogger(__name__)
enable_fp32 = False


def _first_nonfinite_trainable(module, use_grad=False):
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.grad if use_grad else parameter
        if value is not None and not torch.isfinite(value).all():
            return name
    return None


def _first_tensor_device(value):
    if torch.is_tensor(value):
        return value.device
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return _first_tensor_device(item)
            except ValueError:
                pass
    if isinstance(value, dict):
        for item in value.values():
            try:
                return _first_tensor_device(item)
            except ValueError:
                pass
    raise ValueError("No tensor found in reconstruction cache")


def _first_tensor_dtype(value):
    if torch.is_tensor(value):
        return value.dtype
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return _first_tensor_dtype(item)
            except ValueError:
                pass
    if isinstance(value, dict):
        for item in value.values():
            try:
                return _first_tensor_dtype(item)
            except ValueError:
                pass
    raise ValueError("No tensor found in reconstruction cache")


def _index_on_tensor_device(index, tensor):
    """Return an index tensor usable by ``tensor`` without changing values."""
    if torch.is_tensor(index) and index.device != tensor.device:
        return index.to(tensor.device)
    return index


def _select_to_device(tensor, index, device, requires_grad=False):
    """Index a cache tensor locally, then move only the selection to device."""
    selected = tensor[_index_on_tensor_device(index, tensor)].to(device)
    if requires_grad:
        selected.requires_grad_()
    return selected


def _nested_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_nested_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_nested_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _nested_to_device(item, device) for key, item in value.items()}
    return value


def _select_nested_to_device(value, index, device):
    """Select from either a tensor cache or OpenSora's list-valued metadata."""
    if torch.is_tensor(value):
        return _select_to_device(value, index, device)
    if isinstance(value, (list, tuple)):
        scalar_index = int(index.item()) if torch.is_tensor(index) else int(index)
        return _nested_to_device(value[scalar_index], device)
    raise TypeError(f"Unsupported reconstruction cache entry: {type(value)!r}")


def _cat_selected_to_device(tensors, indices, device, requires_grad=False):
    """Select and concatenate cache entries before transferring the mini-batch."""
    selected = torch.cat([
        tensor[_index_on_tensor_device(index, tensor)]
        for tensor, index in zip(tensors, indices)
    ]).to(device)
    if requires_grad:
        selected.requires_grad_()
    return selected


def _forward_reconstruction_block(block, cur_inp):
    if isinstance(cur_inp, tuple):
        if len(cur_inp) > 3:
            if enable_fp32:
                with autocast():
                    return block(
                        cur_inp[0], cur_inp[1], cur_inp[2], cur_inp[3], cur_inp[4]
                    )
            return block(
                cur_inp[0], cur_inp[1], cur_inp[2], cur_inp[3], cur_inp[4]
            )
        if len(cur_inp) == 3:
            if enable_fp32:
                with autocast():
                    return block(cur_inp[0], cur_inp[1], cur_inp[2])
            return block(cur_inp[0], cur_inp[1], cur_inp[2])
        return block(cur_inp[0], cur_inp[1])
    return block(cur_inp)


def _gradient_category(name):
    lowered = name.lower()
    if "delta" in lowered:
        return "delta"
    if "lora" in lowered:
        return "lora"
    if "mask" in lowered:
        return "mask"
    return "other"


def _run_paired_gradient_probe(
    block,
    optimizer,
    cur_inp,
    cur_out,
    cur_grad,
    loss_func,
    ordinary_output,
    scale,
    output_path,
):
    """Compare ordinary FP16 backward with loss-scaled backward.

    The ordinary gradients are restored before returning, so the formal
    optimizer trajectory is unchanged by this diagnostic.
    """
    ordinary_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in block.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    saved_count = loss_func.count
    saved_components = dict(getattr(loss_func, "last_components", {}))

    optimizer.zero_grad(set_to_none=True)
    scaled_output = _forward_reconstruction_block(block, cur_inp)
    scaled_loss = loss_func(scaled_output, cur_out, cur_grad)
    (scaled_loss * scale).backward()

    categories = {}
    per_parameter = []
    for name, parameter in block.named_parameters():
        if name not in ordinary_gradients or parameter.grad is None:
            continue
        ordinary = ordinary_gradients[name].float()
        recovered = parameter.grad.detach().float() / scale
        finite = torch.isfinite(recovered)
        ordinary_finite = torch.isfinite(ordinary)
        both_finite = finite & ordinary_finite
        if both_finite.any():
            ordinary_value = ordinary[both_finite]
            recovered_value = recovered[both_finite]
            ordinary_norm = torch.linalg.vector_norm(ordinary_value)
            recovered_norm = torch.linalg.vector_norm(recovered_value)
            difference_norm = torch.linalg.vector_norm(
                ordinary_value - recovered_value
            )
            relative_difference = float(
                difference_norm / (recovered_norm + 1.0e-30)
            )
            cosine = float(
                (ordinary_value.flatten() @ recovered_value.flatten())
                / (ordinary_norm * recovered_norm + 1.0e-30)
            )
        else:
            ordinary_norm = torch.tensor(0.0)
            recovered_norm = torch.tensor(0.0)
            difference_norm = torch.tensor(0.0)
            relative_difference = float("inf")
            cosine = float("nan")
        ordinary_nonzero = ordinary != 0
        recovered_nonzero = recovered != 0
        recovered_from_zero = int((~ordinary_nonzero & recovered_nonzero).sum())
        lost_after_scaling = int((ordinary_nonzero & ~recovered_nonzero).sum())
        record = {
            "name": name,
            "category": _gradient_category(name),
            "numel": int(parameter.numel()),
            "ordinary_norm": float(ordinary_norm),
            "scaled_recovered_norm": float(recovered_norm),
            "difference_norm": float(difference_norm),
            "relative_difference": relative_difference,
            "cosine": cosine,
            "ordinary_zero_count": int((~ordinary_nonzero).sum()),
            "scaled_recovered_zero_count": int((~recovered_nonzero).sum()),
            "recovered_from_zero_count": recovered_from_zero,
            "lost_after_scaling_count": lost_after_scaling,
            "ordinary_nonfinite_count": int((~ordinary_finite).sum()),
            "scaled_nonfinite_count": int((~finite).sum()),
        }
        per_parameter.append(record)
        aggregate = categories.setdefault(record["category"], {
            "numel": 0,
            "ordinary_squared_norm": 0.0,
            "scaled_squared_norm": 0.0,
            "difference_squared_norm": 0.0,
            "ordinary_zero_count": 0,
            "scaled_recovered_zero_count": 0,
            "recovered_from_zero_count": 0,
            "lost_after_scaling_count": 0,
            "ordinary_nonfinite_count": 0,
            "scaled_nonfinite_count": 0,
        })
        aggregate["numel"] += record["numel"]
        aggregate["ordinary_squared_norm"] += record["ordinary_norm"] ** 2
        aggregate["scaled_squared_norm"] += record["scaled_recovered_norm"] ** 2
        aggregate["difference_squared_norm"] += record["difference_norm"] ** 2
        for key in (
            "ordinary_zero_count", "scaled_recovered_zero_count",
            "recovered_from_zero_count", "lost_after_scaling_count",
            "ordinary_nonfinite_count", "scaled_nonfinite_count",
        ):
            aggregate[key] += record[key]

    for aggregate in categories.values():
        ordinary_norm = math.sqrt(aggregate.pop("ordinary_squared_norm"))
        scaled_norm = math.sqrt(aggregate.pop("scaled_squared_norm"))
        difference_norm = math.sqrt(aggregate.pop("difference_squared_norm"))
        aggregate["ordinary_norm"] = ordinary_norm
        aggregate["scaled_recovered_norm"] = scaled_norm
        aggregate["difference_norm"] = difference_norm
        aggregate["relative_difference"] = difference_norm / max(
            scaled_norm, 1.0e-30
        )

    output_difference = torch.linalg.vector_norm(
        ordinary_output.detach().float() - scaled_output.detach().float()
    ) / (torch.linalg.vector_norm(scaled_output.detach().float()) + 1.0e-30)
    per_parameter.sort(
        key=lambda item: (
            math.isfinite(item["relative_difference"]),
            item["relative_difference"],
        ),
        reverse=True,
    )
    record = {
        "scale": float(scale),
        "ordinary_loss": float(saved_components.get("total", float("nan"))),
        "scaled_probe_loss": float(scaled_loss.detach()),
        "output_relative_difference": float(output_difference),
        "categories": categories,
        "top_parameters": per_parameter[:100],
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=True)

    optimizer.zero_grad(set_to_none=True)
    named_parameters = dict(block.named_parameters())
    for name, gradient in ordinary_gradients.items():
        named_parameters[name].grad = gradient
    loss_func.count = saved_count
    loss_func.last_components = saved_components
    logger.info("Paired gradient scale probe written to %s", output_path)


def _component_gradient_statistics(named_parameters, gradients):
    """Aggregate one loss component's gradients without touching .grad."""
    categories = {}
    by_name = {}
    for (name, parameter), gradient in zip(named_parameters, gradients):
        category = _gradient_category(name)
        aggregate = categories.setdefault(category, {
            "squared_norm": 0.0,
            "numel": 0,
            "nonzero": 0,
            "nonfinite": 0,
        })
        aggregate["numel"] += int(parameter.numel())
        if gradient is None:
            by_name[name] = None
            continue
        value = gradient.detach().float()
        finite = torch.isfinite(value)
        aggregate["nonfinite"] += int((~finite).sum())
        value = torch.where(finite, value, torch.zeros_like(value))
        norm = float(torch.linalg.vector_norm(value))
        aggregate["squared_norm"] += norm ** 2
        aggregate["nonzero"] += int((value != 0).sum())
        by_name[name] = value
    for aggregate in categories.values():
        aggregate["norm"] = math.sqrt(aggregate.pop("squared_norm"))
    total_norm = math.sqrt(sum(item["norm"] ** 2 for item in categories.values()))
    return total_norm, categories, by_name


def _run_loss_component_gradient_probe(
    block,
    loss_func,
    iteration,
    grad_scale,
    output_path,
):
    """Measure reconstruction and each weighted MTD term's gradient impact.

    Each component uses a standalone backward pass whose .grad values are
    copied and cleared before the formal total-loss backward.  This works with
    PyTorch's reentrant gradient checkpointing, which rejects autograd.grad.
    In FP16, each component is multiplied by the active GradScaler scale and
    divided back in FP32 afterwards.
    """
    components = getattr(loss_func, "last_component_tensors", {})
    component_names = (
        "reconstruction", "mtd_local", "mtd_motion", "mtd_global"
    )
    component_tensors = {
        name: components.get(name) for name in component_names
    }
    if not all(
        torch.is_tensor(value) and value.requires_grad
        for value in component_tensors.values()
    ):
        logger.warning(
            "Skipping component gradient probe at iteration %d: "
            "loss component tensors are unavailable",
            iteration,
        )
        return

    named_parameters = [
        (name, parameter)
        for name, parameter in block.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [parameter for _, parameter in named_parameters]
    scale = max(float(grad_scale), 1.0)
    scale_retries = 0

    def backward_component(component, component_scale):
        for parameter in parameters:
            parameter.grad = None
        (component * component_scale).backward(retain_graph=True)
        gradients = tuple(
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in parameters
        )
        for parameter in parameters:
            parameter.grad = None
        return gradients

    while True:
        scaled_gradients = {
            name: backward_component(component_tensors[name], scale)
            for name in component_names
        }
        all_finite = all(
            gradient is None or torch.isfinite(gradient).all()
            for gradients in scaled_gradients.values()
            for gradient in gradients
        )
        if all_finite or scale <= 1.0:
            break
        scale = max(scale / 2.0, 1.0)
        scale_retries += 1
    # Convert to FP32 before unscaling.  Dividing in FP16 would reintroduce
    # underflow and make small but valid component gradients appear as zero.
    gradients_by_component = {
        name: tuple(
            None if gradient is None
            else gradient.detach().float() / scale
            for gradient in gradients
        )
        for name, gradients in scaled_gradients.items()
    }

    # The aggregate MTD gradient is exactly the sum of the three already
    # weighted component gradients; avoid an additional backward pass.
    mtd_gradients = []
    for gradients in zip(
        gradients_by_component["mtd_local"],
        gradients_by_component["mtd_motion"],
        gradients_by_component["mtd_global"],
    ):
        values = [gradient for gradient in gradients if gradient is not None]
        mtd_gradients.append(sum(values) if values else None)
    gradients_by_component["mtd"] = tuple(mtd_gradients)

    statistics_by_component = {}
    for name, gradients in gradients_by_component.items():
        statistics_by_component[name] = _component_gradient_statistics(
            named_parameters, gradients
        )

    rec_norm, rec_categories, rec_by_name = statistics_by_component[
        "reconstruction"
    ]
    mtd_norm, mtd_categories, mtd_by_name = statistics_by_component["mtd"]

    def alignment_with_reconstruction(name):
        norm, categories, by_name = statistics_by_component[name]
        dot = 0.0
        for parameter_name, _ in named_parameters:
            rec_value = rec_by_name.get(parameter_name)
            value = by_name.get(parameter_name)
            if rec_value is not None and value is not None:
                dot += float(rec_value.flatten() @ value.flatten())
        return {
            "loss": float(component_tensors[name].detach()),
            "grad_norm": norm,
            "ratio_to_rec": norm / max(rec_norm, 1.0e-30),
            "dot_with_rec": dot,
            "cosine_with_rec": dot / max(rec_norm * norm, 1.0e-30),
            "categories": categories,
        }

    subterms = {
        short_name: alignment_with_reconstruction(tensor_name)
        for short_name, tensor_name in (
            ("local", "mtd_local"),
            ("motion", "mtd_motion"),
            ("global", "mtd_global"),
        )
    }

    dot = 0.0
    per_parameter = []
    for name, parameter in named_parameters:
        rec_value = rec_by_name.get(name)
        mtd_value = mtd_by_name.get(name)
        if rec_value is None or mtd_value is None:
            continue
        parameter_dot = float((rec_value.flatten() @ mtd_value.flatten()))
        rec_parameter_norm = float(torch.linalg.vector_norm(rec_value))
        mtd_parameter_norm = float(torch.linalg.vector_norm(mtd_value))
        dot += parameter_dot
        per_parameter.append({
            "name": name,
            "category": _gradient_category(name),
            "rec_norm": rec_parameter_norm,
            "mtd_norm": mtd_parameter_norm,
            "mtd_to_rec_ratio": mtd_parameter_norm / max(rec_parameter_norm, 1.0e-30),
            "cosine": parameter_dot / max(
                rec_parameter_norm * mtd_parameter_norm, 1.0e-30
            ),
        })

    cosine = dot / max(rec_norm * mtd_norm, 1.0e-30)
    per_parameter.sort(key=lambda item: item["mtd_norm"], reverse=True)
    record = {
        "iteration": int(iteration),
        "loss": {
            "reconstruction": float(
                component_tensors["reconstruction"].detach()
            ),
            "mtd": float(sum(
                component_tensors[name]
                for name in ("mtd_local", "mtd_motion", "mtd_global")
            ).detach()),
            "mtd_to_rec_ratio": float(sum(
                component_tensors[name]
                for name in ("mtd_local", "mtd_motion", "mtd_global")
            ).detach()) / max(
                float(component_tensors["reconstruction"].detach()), 1.0e-30
            ),
        },
        "gradient": {
            "scale_used": scale,
            "scale_retries": scale_retries,
            "rec_norm": rec_norm,
            "mtd_norm": mtd_norm,
            "mtd_to_rec_ratio": mtd_norm / max(rec_norm, 1.0e-30),
            "dot": dot,
            "cosine": cosine,
        },
        "categories": {
            "reconstruction": rec_categories,
            "mtd": mtd_categories,
        },
        "subterms": subterms,
        "top_parameters_by_mtd_norm": per_parameter[:100],
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")
    logger.info(
        "Component gradient probe iteration=%d rec_norm=%.6e "
        "mtd_norm=%.6e ratio=%.4f cosine=%.4f "
        "local(r=%.4f,c=%.4f) motion(r=%.4f,c=%.4f) "
        "global(r=%.4f,c=%.4f)",
        iteration, rec_norm, mtd_norm,
        record["gradient"]["mtd_to_rec_ratio"], cosine,
        subterms["local"]["ratio_to_rec"],
        subterms["local"]["cosine_with_rec"],
        subterms["motion"]["ratio_to_rec"],
        subterms["motion"]["cosine_with_rec"],
        subterms["global"]["ratio_to_rec"],
        subterms["global"]["cosine_with_rec"],
    )


def mv_to_gpu(l_x, device='cuda'):
    if l_x is None:
        pass
    elif isinstance(l_x, list):
        new_l_x = []
        for x in l_x:
            if x is None:
                new_l_x.append(x)
            else:
                new_l_x.append(x.to(device))
        l_x = new_l_x
    elif isinstance(l_x, torch.Tensor):
        l_x = l_x.to(device)
    else:
        import ipdb; ipdb.set_trace()
    return l_x


def block_reconstruction(model: QuantModel, block: BaseQuantBlock, calib_data: torch.Tensor, config, param_types, opt_target):
                         # batch_size: int = 32, iters: int = 20000, weight: float = 0.01, opt_mode: str = 'mse',
                         # asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
                         # warmup: float = 0.0, act_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
                         # multi_gpu: bool = False, cond: bool = False, is_sm: bool = False):
    """
    Block reconstruction to optimize the output from each block.

    :param model: QuantModel
    :param block: BaseQuantBlock that needs to be optimized
    :param calib_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param act_quant: use activation quantization or not.
    :param lr: learning rate for act delta learning
    :param p: L_p norm minimization
    :param multi_gpu: use multi-GPU or not, if enabled, we should sync the gradients
    :param cond: conditional generation or not
    :param is_sm: avoid OOM when caching n^2 attention matrix when n is large
    """

    device = model.device
    batch_size = config.calib_data.batch_size
    mark_memory("reconstruction_cache_start", model=model, calib_data=calib_data)

    if len(calib_data)==4:
        if config.model.model_type == 'pixart' or config.model.model_type == 'opensora':
            cached_inps, cached_outs = save_in_out_data(model, block, calib_data, config, model_type=config.model.model_type)
        else:
            assert config.model.model_type == 'sdxl'
            cached_inps, cached_outs = save_in_out_data(model, block, calib_data, config, model_type='sdxl')
    else:
        assert config.model.model_type == 'sd'
        cached_inps, cached_outs = save_in_out_data(model, block, calib_data, config, model_type='sd')
    mark_memory("reconstruction_cache_complete", model=model, cached_inps=cached_inps, cached_outs=cached_outs)
    # cached_inps = mv_to_gpu(cached_inps, device=device)
    # cached_outs = mv_to_gpu(cached_outs, device=device)

    # INFO: get the grad (not supported)
    if opt_target == 'weight_and_activation':
        use_grad = config.quant.weight.optimization.use_grad
    else:
        use_grad = getattr(config.quant, opt_target).optimization.use_grad
    assert not use_grad, "not supported for now"
    if not use_grad:
        cached_grads = None
    else:
        # INFO: does not support for now
        raise NotImplementedError
        cached_grads = save_grad_data(model, block, calib_data, act_quant=False, batch_size=batch_size)  # TODO: reduce act_quant
        cached_grads = cached_grads.to(device)

    # INFO: set the quant states, set_quant_state in SaveData
    # model_quant_weight, model_quant_act = model.get_quant_state()
    # block_quant_weight, block_quant_act = block.get_quant_state()
    # model.set_quant_state(False, False)

    # INFO: setup quant_params and optimizer, use independent lr for each param group
    # DEBUG: currently block_recon only support non-softmax quant_param opt
    opt_params = []  # the param group
    param_group_names = []
    if opt_target == 'weight_and_activation':
        # INFO: should have both of the param groups
        for param_type in param_types['weight']:
            name_ = f"weight.{param_type}"
            param_group_names.append(name_)
            params_ = []
            # INFO: iter through all block modules to get all weight_quantizers
            for layer_name, layer_ in block.named_modules():
                if isinstance(layer_, QuantLayer):
                    params_ += [getattr(layer_.weight_quantizer, param_type)]
                    if layer_.split != 0:
                        params_ += [getattr(layer_.weight_quantizer_0, param_type)]
            opt_params += [{
                'params': params_,
                'lr': getattr(config.quant.weight.optimization.params, param_type).lr,
                }]
        for param_type in param_types['activation']:
            # INFO: iter through all block modules to get all weight_quantizers
            name_ = f"activation.{param_type}"
            param_group_names.append(name_)
            params_ = []
            for layer_name, layer_ in block.named_modules():
                if isinstance(layer_, QuantLayer):
                    params_ = [getattr(layer_.act_quantizer, param_type)]
                    if layer_.split != 0:
                        params_ = [getattr(layer_.act_quantizer_0, param_type)]
            # INFO: a few other layers
            opt_params += [{
                    'params': params_,
                    'lr': getattr(config.quant.activation.optimization.params, param_type).lr,
                    }]

    elif opt_target in ['weight','activation']:
        for param_type in param_types:
            if opt_target == 'weight':
                name_ = f"weight.{param_type}"
                param_group_names.append(name_)
                params_ = []
                # INFO: iter through all block modules to get all weight_quantizers
                for layer_name, layer_ in block.named_modules():
                    if isinstance(layer_, QuantLayer):
                        if getattr(layer_.weight_quantizer, param_type) is None:
                            continue
                        params_ += [getattr(layer_.weight_quantizer, param_type)]
                        if layer_.split != 0:
                            params_ += [getattr(layer_.weight_quantizer_0, param_type)]
                        if layer_.weight_quantizer.round_mode == 'learned_hard_sigmoid':
                            layer_.weight_quantizer.soft_targets = True
                opt_params += [{
                    'params': params_,
                    'lr': getattr(config.quant.weight.optimization.params, param_type).lr,
                    }]
            elif opt_target == 'activation':
                # INFO: iter through all block modules to get all weight_quantizers
                name_ = f"activation.{param_type}"
                param_group_names.append(name_)
                params_ = []
                for layer_name, layer_ in block.named_modules():
                    if isinstance(layer_, QuantLayer):
                        params_ = [getattr(layer_.act_quantizer, param_type)]
                        if layer_.split != 0:
                            params_ = [getattr(layer_.act_quantizer_0, param_type)]
                # INFO: a few other layers
                opt_params += [{
                        'params': params_,
                        'lr': getattr(config.quant.activation.optimization.params, param_type).lr,
                        }]
        params_ = []
        for layer_name, layer_ in block.named_modules():
            if isinstance(layer_, QuantLayer):
                '''optim_flag = True
                for module_name in block.fp_layer_list:
                    if module_name in layer_name:
                        optim_flag = False
                        break
                if not optim_flag:
                    continue'''
                # params_ += [param for name, param in layer_.named_parameters() if 'lora' in name]
                if isinstance(layer_, QuantTemporalAttnLinear):
                    params_ = [param for name, param in layer_.named_parameters() if ('lora' in name and 'minus' not in name) or 'mask' in name]
                else:
                    params_ = [param for name, param in layer_.named_parameters() if ('lora' in name and 'minus' not in name)]
                if layer_.weight_quantizer.delta is None:
                    continue
                # avg_delta = torch.sum(layer_.weight_quantizer.delta) / torch.numel(layer_.weight_quantizer.delta)
                opt_params += [{
                    'params': params_,
                    'lr': 1.e-5,
                    }]
    else:
        raise NotImplementedError

    # optimizer = torch.optim.Adam(opt_params)
    if enable_fp32:
        optimizer = torch.optim.AdamW(opt_params)
    else:
        optimizer = torch.optim.AdamW(opt_params)
    mark_memory("reconstruction_optimizer_created", model=model, optimizer=optimizer, cached_inps=cached_inps, cached_outs=cached_outs)

    if opt_target == 'weight_and_activation':
        iters = config.quant.weight.optimization.iters
        assert config.quant.weight.optimization.iters == config.quant.activation.optimization.iters
        optimization_config = config.quant.weight.optimization
    else:
        optimization_config = getattr(config.quant, opt_target).optimization
        iters = optimization_config.iters
    checkpoint_interval = int(
        getattr(optimization_config, 'checkpoint_interval', 0) or 0
    )
    checkpoint_dir = getattr(config, 'reconstruction_checkpoint_dir', None)
    resume_checkpoint = getattr(config, 'resume_reconstruction', None)
    use_grad_scaler = bool(getattr(config, 'use_grad_scaler', False))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)
    # scheduler = None

    # INFO: unpack the config for loss 
    if opt_target == 'weight_and_activation':
        logging.info("When joint optimization, use weight's quant config")
        config_loss = config.quant.weight.optimization.loss
        config_loss['iters'] = config.quant.weight.optimization.iters
    else:
        config_loss = getattr(config.quant, opt_target).optimization.loss
        config_loss['iters'] = getattr(config.quant, opt_target).optimization.iters
    config_loss['iters'] = config_loss['iters']*0.9  # INFO: anneal to minimum value with 0.7 iters
    config_loss['module_type'] = 'block'
    config_loss['use_reconstruction_loss'] = ('delta' in param_types or 'delta_out' in param_types)
    config_loss['use_round_loss'] = 'alpha' in param_types
    config_loss['mtd_config'] = config
    loss_func = LossFunction(block, **config_loss)

    mtd_config = normalize_mtd_config(config)
    reconstruction_signature = {
        'mtd_enabled': mtd_config['enabled'],
        'noise_channels': mtd_config['noise_channels'],
        'transport_size': mtd_config['transport_size'],
        'temperature': mtd_config['temperature'],
        'search_mode': mtd_config['search_mode'],
        'anchor_radius': mtd_config['anchor_radius'],
        'total_weight': mtd_config['total_weight'],
        'local_transport_weight': mtd_config['local_transport_weight'],
        'motion_residual_weight': mtd_config['motion_residual_weight'],
        'global_relation_weight': mtd_config['global_relation_weight'],
        'batch_size': int(config.calib_data.batch_size),
        'n_steps': int(config.calib_data.n_steps),
        'n_samples': int(config.calib_data.n_samples),
        'weight_bits': int(config.quant.weight.quantizer.n_bits),
        'activation_bits': int(config.quant.activation.quantizer.n_bits),
    }

    # Preserve the original CUDA RNG stream for the sampling plan even when
    # the immutable reconstruction cache resides on CPU.  Generating these
    # indices with the CPU RNG would silently alter the optimization trajectory.
    sampling_device = device
    # sample_idxs = torch.randint(low=0,high=cached_inps.shape[0],size=(iters,batch_size))
    if isinstance(cached_inps, list):
        if isinstance(cached_outs, list):
            idxs_list = []
            for i in range(len(cached_outs)):
                idxs_list.append(torch.randint(low=0,high=cached_inps[1][i].shape[0],size=(iters,1), device=sampling_device))
            pmp_idxs = torch.randint(low=0,high=len(cached_outs),size=(iters, 1), device=sampling_device)
        else:
            sample_idxs = torch.randint(low=0,high=cached_inps[0].shape[0],size=(iters,batch_size), device=sampling_device)
    else:
        sample_idxs = torch.randint(low=0,high=cached_inps.shape[0],size=(iters,batch_size), device=sampling_device)
    if isinstance(cached_outs, list):
        sampling_plan = {
            'kind': 'pmp',
            'pmp_idxs': pmp_idxs,
            'idxs_list': idxs_list,
        }
    else:
        sampling_plan = {'kind': 'sample_idxs', 'sample_idxs': sample_idxs}
    torch.set_grad_enabled(True)
    # import ipdb; ipdb.set_trace()
    # iters = 16 # debug
    scaler = GradScaler(enabled=use_grad_scaler)
    for name, param in block.named_parameters():
        if ('lora' in name and 'minus' not in name) or 'delta' in name or 'mask' in name:
        # if 'lora' in name or 'zero_point' in name or 'delta' in name or 'zp_list' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    current_optimizer_parameter_names = optimizer_parameter_names(block, optimizer)
    start_iteration = 0
    if resume_checkpoint:
        resume_state = torch.load(resume_checkpoint, map_location='cpu')
        if int(resume_state.get('format_version', 0)) != 1:
            raise ValueError(
                f"Unsupported reconstruction checkpoint: {resume_checkpoint}"
            )
        if int(resume_state['total_iterations']) != int(iters):
            raise ValueError(
                f"Checkpoint expects {resume_state['total_iterations']} iterations, "
                f"config requests {iters}"
            )
        if (
            resume_state['opt_target'] != opt_target
            or list(resume_state['param_types']) != list(param_types)
        ):
            raise ValueError(
                "Reconstruction checkpoint optimization target does not match the config"
            )
        saved_signature = dict(resume_state['reconstruction_signature'])
        # Backward compatibility for checkpoints created before total_weight
        # was introduced; their effective global MTD weight was 1.0.
        saved_signature.setdefault('total_weight', 1.0)
        saved_signature.setdefault('search_mode', 'fixed_3x3')
        saved_signature.setdefault('anchor_radius', 4)
        if saved_signature != reconstruction_signature:
            raise ValueError(
                "Reconstruction checkpoint MTD/calibration signature does not match the config"
            )
        if (
            resume_state['optimizer_parameter_names']
            != current_optimizer_parameter_names
        ):
            raise ValueError(
                "Reconstruction checkpoint optimizer parameter ordering does not match the model"
            )
        restore_trainable_parameter_state(
            block, resume_state['trainable_parameters']
        )
        optimizer.load_state_dict(resume_state['optimizer'])
        scheduler.load_state_dict(resume_state['scheduler'])
        if use_grad_scaler and resume_state.get('scaler') is not None:
            scaler.load_state_dict(resume_state['scaler'])
        sampling_plan = nested_tensors_to_device(
            resume_state['sampling_plan'], sampling_device
        )
        if sampling_plan['kind'] == 'pmp':
            pmp_idxs = sampling_plan['pmp_idxs']
            idxs_list = sampling_plan['idxs_list']
        else:
            sample_idxs = sampling_plan['sample_idxs']
        start_iteration = int(resume_state['iteration'])
        if start_iteration < 0 or start_iteration > iters:
            raise ValueError(
                f"Invalid resume iteration {start_iteration} for {iters} total iterations"
            )
        torch.set_rng_state(resume_state['torch_rng_state'])
        random.setstate(resume_state['python_rng_state'])
        np.random.set_state(resume_state['numpy_rng_state'])
        if (
            torch.cuda.is_available()
            and resume_state.get('cuda_rng_state') is not None
        ):
            torch.cuda.set_rng_state(
                resume_state['cuda_rng_state'], device=device
            )
        loss_func.count = start_iteration
        logger.info(
            "Resuming reconstruction from iteration %d/%d using %s",
            start_iteration, iters, resume_checkpoint,
        )
    monitor_interval = int(getattr(config, 'numeric_monitor_interval', 0) or 0)
    numeric_monitor = None
    if monitor_interval > 0 and checkpoint_dir:
        numeric_monitor = NumericMonitor(
            os.path.join(checkpoint_dir, 'numeric_monitor.jsonl'),
            interval=monitor_interval,
            detailed_interval=int(
                getattr(config, 'numeric_monitor_detailed_interval', 100) or 100
            ),
        )
        logger.info(
            "Numerical monitor enabled: interval=%d, detailed_interval=%d, output=%s",
            numeric_monitor.interval,
            numeric_monitor.detailed_interval,
            numeric_monitor.output_path,
        )
    logger.info(
        "Reconstruction precision: model/input dtype=%s, GradScaler=%s",
        _first_tensor_dtype(cached_inps), use_grad_scaler,
    )
    component_probe_interval = int(
        getattr(config, "component_gradient_probe_interval", 0) or 0
    )
    component_probe_iterations = {
        int(iteration)
        for iteration in (
            getattr(config, "component_gradient_probe_iterations", []) or []
        )
    }
    if component_probe_iterations:
        logger.info(
            "MTD subterm gradient probes scheduled at iterations: %s",
            sorted(component_probe_iterations),
        )

    # for name, param in block.named_parameters():
        # print(f"Parameter {name} requires_grad: {param.requires_grad}")

    for i in range(1, 27):
        set_grad_checkpoint(block.blocks[i])

    for i in range(start_iteration, iters):
        # print(i)
        # import time
        # t0 = time.time()
        # idx = torch.randperm(cached_inps.size(0))[:batch_size]
        if isinstance(cached_outs, list):
            pmp_id = int(pmp_idxs[i].item())
            idx = idxs_list[pmp_id][i]
        else:
            idx = sample_idxs[i,:]
        # import ipdb; ipdb.set_trace()
        if isinstance(cached_inps, list):
            # 这个对应多输入
            if len(cached_inps)==2:
                # idx = torch.randperm(cached_inps[0].size(0))[:batch_size]
                cur_x = _select_to_device(cached_inps[0], idx, device)
                cur_t = _select_to_device(cached_inps[1], idx, device)
                cur_inp = (cur_x, cur_t)
            elif len(cached_inps)==3:
                # idx = torch.randperm(cached_inps[0].size(0))[:batch_size]
                cur_x = _select_to_device(cached_inps[0], idx, device)
                cur_t = _select_to_device(cached_inps[1], idx, device)
                cur_y = _select_to_device(cached_inps[2], idx, device)
                cur_inp = (cur_x, cur_t, cur_y)
            else:
                # 针对 QuantTransformerBlock
                cur_inp = []
                # idx = torch.randperm(cached_inps[0].size(0))[:batch_size]
                for j in range(len(cached_inps)):
                    if j in [1]:
                        cur_inp.append(_select_to_device(
                            cached_inps[j][pmp_id], idx, device, requires_grad=True
                        ))
                    elif j in [4]:
                        # 4 prob is None
                        if cached_inps[4] is None:
                            cur_inp.append(None)
                        else:
                            cur_inp.append(_select_nested_to_device(
                                cached_inps[j][pmp_id], idx, device
                            ))
                    elif j in [3]:
                        cur_inp.append(_select_nested_to_device(
                            cached_inps[j][pmp_id], idx, device
                        ))
                    else:
                        cur_inp.append(_cat_selected_to_device(
                            [cached_inps[j][pmp_id]] * 4,
                            [idx*4, idx*4+1, idx*4+2, idx*4+3],
                            device,
                            requires_grad=True,
                        ))
                    '''if cached_inps[j] == None:
                        cur_inp.append(None)
                    else:
                        cur_inp.append(cached_inps[j][idx])'''
                
                cur_inp = tuple(cur_inp)
        else:
            # idx = torch.randperm(cached_inps.size(0))[:batch_size]  # 随机取样
            cur_inp = _select_to_device(cached_inps, idx, device)
        if isinstance(cached_outs, list):
            # cur_out = cached_outs[pmp_id][idx]
            cur_out = _cat_selected_to_device(
                [cached_outs[pmp_id]] * 4,
                [idx*4, idx*4+1, idx*4+2, idx*4+3],
                device,
            )
        else:
            cur_out = _select_to_device(cached_outs, idx, device)
        cur_grad = _select_to_device(cached_grads, idx, device) if use_grad else None

        # import ipdb; ipdb.set_trace()
        optimizer.zero_grad()
        # cur_inp.requires_grad_()
        out_quant = _forward_reconstruction_block(block, cur_inp)
        if i == 0:
            mark_memory("reconstruction_iter_1_after_forward", model=model, optimizer=optimizer, cached_inps=cached_inps, cached_outs=cached_outs, out_quant=out_quant)

        # t2 = time.time()
        # logger.info('infer time {}'.format(t2 - t1))
        # import ipdb; ipdb.set_trace()
        err = loss_func(out_quant, cur_out, cur_grad)
        # t3 = time.time()
        # logger.info('loss time {}'.format(t3 - t2))
        # check nan
        
        if not torch.isfinite(err):
            raise FloatingPointError(
                f"Non-finite reconstruction loss at iteration {i + 1}"
            )
        completed_iterations = i + 1
        should_component_probe = (
            checkpoint_dir
            and (
                completed_iterations in component_probe_iterations
                or (
                    component_probe_interval > 0
                    and (
                        i == start_iteration
                        or completed_iterations % component_probe_interval == 0
                        or completed_iterations == iters
                    )
                )
            )
        )
        if should_component_probe:
            _run_loss_component_gradient_probe(
                block=block,
                loss_func=loss_func,
                iteration=completed_iterations,
                grad_scale=(scaler.get_scale() if use_grad_scaler else 1.0),
                output_path=os.path.join(
                    checkpoint_dir, "loss_component_gradients.jsonl"
                ),
            )
        if use_grad_scaler:
            scaler.scale(err).backward()
            scaler.unscale_(optimizer)
        else:
            err.backward()  # DEBUG_ONLY: cancel retrain_graph
        # Do not retain graph-connected loss tensors after backward.
        loss_func.last_component_tensors = {}
        paired_probe_scale = float(
            getattr(config, "paired_gradient_probe_scale", 0.0) or 0.0
        )
        if (
            i == start_iteration
            and paired_probe_scale > 1.0
            and not use_grad_scaler
            and checkpoint_dir
        ):
            _run_paired_gradient_probe(
                block=block,
                optimizer=optimizer,
                cur_inp=cur_inp,
                cur_out=cur_out,
                cur_grad=cur_grad,
                loss_func=loss_func,
                ordinary_output=out_quant,
                scale=paired_probe_scale,
                output_path=os.path.join(
                    checkpoint_dir, "paired_gradient_probe.json"
                ),
            )
        if i < start_iteration + 3:
            bad_gradient = _first_nonfinite_trainable(block, use_grad=True)
            if bad_gradient is not None:
                if use_grad_scaler:
                    logger.warning(
                        'GradScaler detected a non-finite gradient in "%s" '
                        'at iteration %d; this optimizer update will be skipped '
                        'and the scale reduced.',
                        bad_gradient, i + 1,
                    )
                else:
                    raise FloatingPointError(
                        f'Non-finite gradient in "{bad_gradient}" at iteration {i + 1}'
                    )
        if i == 0:
            mark_memory("reconstruction_iter_1_after_backward", model=model, optimizer=optimizer, cached_inps=cached_inps, cached_outs=cached_outs)
        # err.backward(retain_graph=True)
        # t4  = time.time()
        # logger.info('backward time {}'.format(t4 - t3))

        # if multi_gpu:
            # raise NotImplementedError
            # for p in opt_params:
            #     link.allreduce(p.grad)
        if numeric_monitor is not None:
            numeric_monitor.record_iteration(
                completed_iterations,
                block,
                optimizer,
                out_quant,
                cur_out,
                err,
                loss_components=getattr(loss_func, 'last_components', None),
                scaler=scaler,
            )

        if use_grad_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if i < start_iteration + 3:
            bad_parameter = _first_nonfinite_trainable(block, use_grad=False)
            if bad_parameter is not None:
                raise FloatingPointError(
                    f'Non-finite parameter in "{bad_parameter}" after iteration {i + 1}'
                )
        if scheduler:
            scheduler.step()
        should_checkpoint = (
            checkpoint_dir
            and checkpoint_interval > 0
            and (
                completed_iterations % checkpoint_interval == 0
                or completed_iterations == iters
            )
        )
        if should_checkpoint:
            training_state = {
                'format_version': 1,
                'iteration': completed_iterations,
                'total_iterations': int(iters),
                'opt_target': opt_target,
                'param_types': list(param_types),
                'reconstruction_signature': reconstruction_signature,
                'optimizer_parameter_names': current_optimizer_parameter_names,
                'trainable_parameters': trainable_parameter_state(block),
                'optimizer': nested_tensors_to_cpu(optimizer.state_dict()),
                'scheduler': (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                'scaler': scaler.state_dict() if use_grad_scaler else None,
                'sampling_plan': nested_tensors_to_cpu(sampling_plan),
                'torch_rng_state': torch.get_rng_state(),
                'python_rng_state': random.getstate(),
                'numpy_rng_state': np.random.get_state(),
                'cuda_rng_state': (
                    torch.cuda.get_rng_state(device)
                    if torch.cuda.is_available() else None
                ),
            }
            resume_path = os.path.join(
                checkpoint_dir, 'reconstruction_state_latest.pth'
            )
            iteration_resume_path = os.path.join(
                checkpoint_dir,
                f'reconstruction_state_iter_{completed_iterations:08d}.pth',
            )
            inference_path = os.path.join(
                checkpoint_dir,
                f'ckpt_iter_{completed_iterations:08d}.pth',
            )
            atomic_torch_save(training_state, resume_path)
            atomic_torch_save(training_state, iteration_resume_path)
            atomic_torch_save(
                model.get_inference_quant_params_dict(), inference_path
            )
            logger.info(
                "Saved reconstruction state and inference checkpoint at "
                "iteration %d: %s, %s, %s",
                completed_iterations, resume_path, iteration_resume_path,
                inference_path,
            )
        checkpoints = {1, max(1, iters // 4), max(1, iters // 2), max(1, 3 * iters // 4), iters}
        if i + 1 in checkpoints:
            mark_memory(f"reconstruction_iter_{i + 1}_complete", model=model, optimizer=optimizer, cached_inps=cached_inps, cached_outs=cached_outs)

    # import ipdb; ipdb.set_trace()
    torch.cuda.empty_cache()
    mark_memory("reconstruction_cleanup", model=model, optimizer=optimizer)

    # Finish optimization, use hard rounding.
    for layer_name, layer_ in block.named_modules():
        if isinstance(layer_, QuantLayer):
            if layer_.weight_quantizer.round_mode == 'learned_hard_sigmoid':
                layer_.weight_quantizer.soft_targets = False
    # DEBUG: should not always use
    # layer.weight_quantizer.soft_targets = False
    # if layer.split != 0:
        # layer.weight_quantizer_0.soft_targets = False

    return None
