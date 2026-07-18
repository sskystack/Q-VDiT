import torch
import os
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
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
from qdiff.research import normalize_research_config, sample_trajectory_pair_indices, set_trajectory_position
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

logger = logging.getLogger(__name__)
enable_fp32 = False


def _pin_cpu_tensors(obj):
    """Copy a nested CPU batch into page-locked staging memory."""
    if torch.is_tensor(obj):
        if obj.device.type == 'cpu' and not obj.is_pinned():
            return obj.pin_memory()
        return obj
    if isinstance(obj, tuple):
        return tuple(_pin_cpu_tensors(item) for item in obj)
    if isinstance(obj, list):
        return [_pin_cpu_tensors(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _pin_cpu_tensors(value) for key, value in obj.items()}
    return obj


def _move_to_device(obj, device, non_blocking=False):
    """Move tensors in an arbitrarily nested reconstruction batch."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, tuple):
        return tuple(_move_to_device(item, device, non_blocking) for item in obj)
    if isinstance(obj, list):
        return [_move_to_device(item, device, non_blocking) for item in obj]
    if isinstance(obj, dict):
        return {key: _move_to_device(value, device, non_blocking) for key, value in obj.items()}
    return obj


class _AsyncCudaBatchPrefetcher:
    """One-batch-ahead CPU staging and CUDA-stream prefetch.

    CPU indexing and pinning run in a worker thread while the default stream
    reconstructs the current batch.  The returned host references are kept
    alive until the consumer has completed the corresponding iteration.
    """

    def __init__(self, load_batch, device, pin_memory=True):
        self.load_batch = load_batch
        self.device = torch.device(device)
        self.pin_memory = pin_memory
        self.stream = torch.cuda.Stream(device=self.device)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='qvdit-prefetch')
        self.future = None

    def _prepare(self, iteration):
        batch = self.load_batch(iteration)
        host_batch = _pin_cpu_tensors(batch) if self.pin_memory else batch
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            device_batch = _move_to_device(host_batch, self.device, non_blocking=self.pin_memory)
            ready = torch.cuda.Event()
            ready.record(self.stream)
        return device_batch, host_batch, ready

    def start(self, iteration=0):
        self.future = self.executor.submit(self._prepare, iteration)

    def next(self, next_iteration=None):
        device_batch, host_batch, ready = self.future.result()
        torch.cuda.current_stream(self.device).wait_event(ready)
        if next_iteration is None:
            self.future = None
        else:
            self.future = self.executor.submit(self._prepare, next_iteration)
        return device_batch, host_batch

    def close(self):
        if self.future is not None:
            self.future.result()
        self.executor.shutdown(wait=True)


def _select_reconstruction_batch(cached_inps, cached_outs, cached_grads, use_grad, idx, pmp_id=None):
    """Select one reconstruction mini-batch without changing cache placement."""
    if isinstance(cached_outs, list):
        pmp_id = int(pmp_id.item()) if torch.is_tensor(pmp_id) else int(pmp_id)

    if isinstance(cached_inps, list):
        if len(cached_inps) == 2:
            cur_inp = (cached_inps[0][idx], cached_inps[1][idx])
        elif len(cached_inps) == 3:
            cur_inp = (cached_inps[0][idx], cached_inps[1][idx], cached_inps[2][idx])
        else:
            selected = []
            scalar_idx = int(idx.item()) if torch.is_tensor(idx) and idx.numel() == 1 else idx
            for j in range(len(cached_inps)):
                if j == 4 and cached_inps[j] is None:
                    selected.append(None)
                elif j in (1, 3, 4):
                    selected.append(cached_inps[j][pmp_id][idx])
                else:
                    selected.append(torch.cat([
                        cached_inps[j][pmp_id][scalar_idx * 4 + offset]
                        for offset in range(4)
                    ]))
            cur_inp = tuple(selected)
    else:
        cur_inp = cached_inps[idx]

    if isinstance(cached_outs, list):
        scalar_idx = int(idx.item()) if torch.is_tensor(idx) else int(idx)
        cur_out = torch.cat([
            cached_outs[pmp_id][scalar_idx * 4 + offset]
            for offset in range(4)
        ])
    else:
        cur_out = cached_outs[idx]
    cur_grad = cached_grads[idx] if use_grad else None
    return cur_inp, cur_out, cur_grad


def _first_nonfinite_trainable(module, use_grad=False):
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.grad if use_grad else parameter
        if value is not None and not torch.isfinite(value).all():
            return name
    return None


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

    if len(calib_data)==4:
        if config.model.model_type == 'pixart' or config.model.model_type == 'opensora':
            cached_inps, cached_outs = save_in_out_data(model, block, calib_data, config, model_type=config.model.model_type)
        else:
            assert config.model.model_type == 'sdxl'
            cached_inps, cached_outs = save_in_out_data(model, block, calib_data, config, model_type='sdxl')
    else:
        assert config.model.model_type == 'sd'
        cached_inps, cached_outs = save_in_out_data(model, block, calib_data, config, model_type='sd')
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
                params_ = [
                    param for name, param in layer_.named_parameters()
                    if ('lora' in name and 'minus' not in name)
                    or 'mask' in name
                    or 'tarq' in name
                    or 'taq_' in name
                ]
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

    if opt_target == 'weight_and_activation':
        iters = config.quant.weight.optimization.iters
        assert config.quant.weight.optimization.iters == config.quant.activation.optimization.iters
        optimization_config = config.quant.weight.optimization
    else:
        optimization_config = getattr(config.quant, opt_target).optimization
        iters = optimization_config.iters
    checkpoint_interval = int(getattr(optimization_config, 'checkpoint_interval', 0) or 0)
    checkpoint_dir = getattr(config, 'reconstruction_checkpoint_dir', None)
    resume_checkpoint = getattr(config, 'resume_reconstruction', None)
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
    config_loss['research_config'] = config
    loss_func = LossFunction(block, **config_loss)

    research_config = normalize_research_config(config)
    reconstruction_signature = {
        'token_axis': research_config['token_axis'],
        'frame_axis': research_config['frame_axis'],
        'diffusion_axis': research_config['diffusion_axis'],
        'batch_size': int(config.calib_data.batch_size),
        'n_steps': int(config.calib_data.n_steps),
        'n_samples': int(config.calib_data.n_samples),
        'weight_bits': int(config.quant.weight.quantizer.n_bits),
        'activation_bits': int(config.quant.activation.quantizer.n_bits),
    }
    taq_enabled = research_config['diffusion_axis'] == 'TAQ'
    keep_cache_on_cpu = bool(getattr(config.calib_data, 'keep_cache_on_cpu', False))
    async_prefetch = bool(getattr(config.calib_data, 'async_prefetch', True))
    pin_memory = bool(getattr(config.calib_data, 'pin_memory', True))

    # move to gpu device
    # sample_idxs = torch.randint(low=0,high=cached_inps.shape[0],size=(iters,batch_size))
    if isinstance(cached_inps, list):
        if isinstance(cached_outs, list):
            idxs_list = []
            for i in range(len(cached_outs)):
                idxs_list.append(torch.randint(low=0,high=cached_inps[1][i].shape[0],size=(iters,1), device=cached_inps[1][i].device))
            pmp_idxs = torch.randint(low=0,high=len(cached_outs),size=(iters, 1), device=cached_inps[0][0].device)
        else:
            if taq_enabled:
                sample_idxs = sample_trajectory_pair_indices(
                    cached_inps[0].shape[0], config.calib_data.n_steps, config.calib_data.n_samples * 2,
                    iters, batch_size, cached_inps[0].device, research_config['taq']['num_bins']
                )
            else:
                sample_idxs = torch.randint(low=0,high=cached_inps[0].shape[0],size=(iters,batch_size), device=cached_inps[0].device)
    else:
        if taq_enabled:
            sample_idxs = sample_trajectory_pair_indices(
                cached_inps.shape[0], config.calib_data.n_steps, config.calib_data.n_samples * 2,
                iters, batch_size, cached_inps.device, research_config['taq']['num_bins']
            )
        else:
            sample_idxs = torch.randint(low=0,high=cached_inps.shape[0],size=(iters,batch_size), device=cached_inps.device)

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
    if enable_fp32:
        scaler = GradScaler()
    for name, param in block.named_parameters():
        if ('lora' in name and 'minus' not in name) or 'delta' in name or 'mask' in name or 'tarq' in name or 'taq_' in name:
        # if 'lora' in name or 'zero_point' in name or 'delta' in name or 'zp_list' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # When a layer carries a TARQ branch its forward skips the TQE output-lora
    # (and the temporal frame mask), so those parameters have no gradient path.
    # Freeze them explicitly instead of letting them sit in the optimizer as
    # permanently zero-gradient entries.
    for module_ in block.modules():
        if getattr(module_, 'tarq', None) is not None:
            module_.loraA_out.weight.requires_grad = False
            module_.loraB_out.weight.requires_grad = False
            mask_param = getattr(module_, 'mask', None)
            if isinstance(mask_param, torch.nn.Parameter):
                mask_param.requires_grad = False

    current_optimizer_parameter_names = optimizer_parameter_names(block, optimizer)
    start_iteration = 0
    if resume_checkpoint:
        resume_state = torch.load(resume_checkpoint, map_location='cpu')
        if int(resume_state.get('format_version', 0)) != 1:
            raise ValueError(f"Unsupported reconstruction checkpoint: {resume_checkpoint}")
        if int(resume_state['total_iterations']) != int(iters):
            raise ValueError(
                f"Checkpoint expects {resume_state['total_iterations']} iterations, config requests {iters}"
            )
        if resume_state['opt_target'] != opt_target or list(resume_state['param_types']) != list(param_types):
            raise ValueError("Reconstruction checkpoint optimization target does not match the config")
        if resume_state['reconstruction_signature'] != reconstruction_signature:
            raise ValueError("Reconstruction checkpoint method/calibration signature does not match the config")
        if resume_state['optimizer_parameter_names'] != current_optimizer_parameter_names:
            raise ValueError("Reconstruction checkpoint optimizer parameter ordering does not match the model")
        restore_trainable_parameter_state(block, resume_state['trainable_parameters'])
        optimizer.load_state_dict(resume_state['optimizer'])
        scheduler.load_state_dict(resume_state['scheduler'])
        if enable_fp32 and resume_state.get('scaler') is not None:
            scaler.load_state_dict(resume_state['scaler'])
        sampling_device = cached_inps[0].device if isinstance(cached_inps, list) else cached_inps.device
        sampling_plan = nested_tensors_to_device(resume_state['sampling_plan'], sampling_device)
        if sampling_plan['kind'] == 'pmp':
            pmp_idxs = sampling_plan['pmp_idxs']
            idxs_list = sampling_plan['idxs_list']
        else:
            sample_idxs = sampling_plan['sample_idxs']
        start_iteration = int(resume_state['iteration'])
        if start_iteration < 0 or start_iteration > iters:
            raise ValueError(f"Invalid resume iteration {start_iteration} for {iters} total iterations")
        torch.set_rng_state(resume_state['torch_rng_state'])
        random.setstate(resume_state['python_rng_state'])
        np.random.set_state(resume_state['numpy_rng_state'])
        if torch.cuda.is_available() and resume_state.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state(resume_state['cuda_rng_state'], device=device)
        loss_func.count = start_iteration
        logger.info(
            "Resuming reconstruction from iteration %d/%d using %s",
            start_iteration, iters, resume_checkpoint,
        )

    # for name, param in block.named_parameters():
        # print(f"Parameter {name} requires_grad: {param.requires_grad}")

    # TARQ attaches trainable transport/gating branches to the first and last
    # transformer blocks as well.  Leaving those two blocks outside gradient
    # checkpointing retains several GiB of their full FP32 graphs and makes a
    # true reconstruction batch of four exceed a 48 GiB device.
    for block_index, transformer_block in enumerate(block.blocks):
        # The first block receives frozen embeddings, so it needs the
        # non-reentrant variant to compute parameter gradients without a
        # grad-requiring input.  Later blocks receive the first block's
        # trainable output and can use the lower-memory legacy variant.
        set_grad_checkpoint(transformer_block, use_reentrant=(block_index != 0))

    def load_reconstruction_batch(iteration):
        if isinstance(cached_outs, list):
            selected_pmp = pmp_idxs[iteration]
            selected_pmp_id = int(selected_pmp.item())
            selected_idx = idxs_list[selected_pmp_id][iteration]
        else:
            selected_pmp = None
            selected_idx = sample_idxs[iteration, :]
        selected = _select_reconstruction_batch(
            cached_inps, cached_outs, cached_grads, use_grad,
            selected_idx, selected_pmp,
        )
        trajectory_step = None
        if taq_enabled and not isinstance(cached_outs, list):
            trajectory_step = int(selected_idx[0].item()) // (config.calib_data.n_samples * 2)
        return (*selected, trajectory_step)

    prefetcher = None
    if start_iteration < iters and keep_cache_on_cpu and async_prefetch and torch.device(device).type == 'cuda':
        logger.info("Using pinned FP32 CPU cache with one-batch-ahead CUDA prefetch")
        prefetcher = _AsyncCudaBatchPrefetcher(load_reconstruction_batch, device, pin_memory=pin_memory)
        prefetcher.start(start_iteration)

    for i in range(start_iteration, iters):
        # print(i)
        # import time
        # t0 = time.time()
        # idx = torch.randperm(cached_inps.size(0))[:batch_size]
        if prefetcher is not None:
            prefetched, host_batch = prefetcher.next(i + 1 if i + 1 < iters else None)
            cur_inp, cur_out, cur_grad, trajectory_step = prefetched
        else:
            cur_inp, cur_out, cur_grad, trajectory_step = load_reconstruction_batch(i)
            if keep_cache_on_cpu:
                cur_inp = _move_to_device(cur_inp, device)
                cur_out = _move_to_device(cur_out, device)
                cur_grad = _move_to_device(cur_grad, device)

        if trajectory_step is not None:
            set_trajectory_position(block, trajectory_step, config.calib_data.n_steps)

        # import ipdb; ipdb.set_trace()
        optimizer.zero_grad()
        # cur_inp.requires_grad_()
        if isinstance(cur_inp, tuple):
            if len(cur_inp) > 3:
                # out_quant = block(cur_inp)  # 目前只针对 QuantTransformerblock，该block有多个输入，这时的输入为元组，包含了原本的所有输入
                if enable_fp32:
                    with autocast():
                        out_quant = block(cur_inp[0], cur_inp[1], cur_inp[2], cur_inp[3], cur_inp[4])
                else:
                    out_quant = block(cur_inp[0], cur_inp[1], cur_inp[2], cur_inp[3], cur_inp[4])
            elif len(cur_inp) == 3:
                if enable_fp32:
                    with autocast():
                        out_quant = block(cur_inp[0], cur_inp[1], cur_inp[2])
                else:
                    out_quant = block(cur_inp[0], cur_inp[1], cur_inp[2])
            else:
                out_quant = block(cur_inp[0], cur_inp[1])
        else:
            out_quant = block(cur_inp)

        # t2 = time.time()
        # logger.info('infer time {}'.format(t2 - t1))
        # import ipdb; ipdb.set_trace()
        err = loss_func(out_quant, cur_out, cur_grad)
        # t3 = time.time()
        # logger.info('loss time {}'.format(t3 - t2))
        # check nan
        
        if not torch.isfinite(err):
            raise FloatingPointError(f"Non-finite reconstruction loss at iteration {i + 1}")
        if enable_fp32:
            scaler.scale(err).backward()
        else:
            err.backward()  # DEBUG_ONLY: cancel retrain_graph
        if i < 3:
            bad_gradient = _first_nonfinite_trainable(block, use_grad=True)
            if bad_gradient is not None:
                raise FloatingPointError(
                    f'Non-finite gradient in "{bad_gradient}" at iteration {i + 1}'
                )
        # err.backward(retain_graph=True)
        # t4  = time.time()
        # logger.info('backward time {}'.format(t4 - t3))

        # if multi_gpu:
            # raise NotImplementedError
            # for p in opt_params:
            #     link.allreduce(p.grad)
        if enable_fp32:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if i < 3:
            bad_parameter = _first_nonfinite_trainable(block, use_grad=False)
            if bad_parameter is not None:
                raise FloatingPointError(
                    f'Non-finite parameter in "{bad_parameter}" after iteration {i + 1}'
                )
        if scheduler:
            scheduler.step()

        completed_iterations = i + 1
        should_checkpoint = (
            checkpoint_dir
            and checkpoint_interval > 0
            and (completed_iterations % checkpoint_interval == 0 or completed_iterations == iters)
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
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'scaler': scaler.state_dict() if enable_fp32 else None,
                'sampling_plan': nested_tensors_to_cpu(sampling_plan),
                'torch_rng_state': torch.get_rng_state(),
                'python_rng_state': random.getstate(),
                'numpy_rng_state': np.random.get_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None,
            }
            resume_path = os.path.join(checkpoint_dir, 'reconstruction_state_latest.pth')
            inference_path = os.path.join(
                checkpoint_dir, f'ckpt_iter_{completed_iterations:08d}.pth'
            )
            atomic_torch_save(training_state, resume_path)
            atomic_torch_save(model.get_inference_quant_params_dict(), inference_path)
            logger.info(
                "Saved reconstruction state and inference checkpoint at iteration %d: %s, %s",
                completed_iterations, resume_path, inference_path,
            )

    if prefetcher is not None:
        prefetcher.close()

    # import ipdb; ipdb.set_trace()
    torch.cuda.empty_cache()

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
