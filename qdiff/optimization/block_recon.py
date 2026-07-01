import os
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
from torch.cuda.amp import GradScaler, autocast
from opensora.acceleration.checkpoint import set_grad_checkpoint

logger = logging.getLogger(__name__)
enable_fp32 = False
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


def index_to_device(x, idx, device):
    return x[idx].to(device)


def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    return x


def sample_cfg_pair_indices(total_size, n_samples, iters, pair_batch_size, device):
    group_size = n_samples * 2
    if group_size <= 0:
        raise ValueError("n_samples must be positive when using CFG residual loss")
    all_indices = torch.arange(total_size, device=device)
    cond_indices = all_indices[(all_indices % group_size) < n_samples]
    if cond_indices.numel() == 0:
        raise ValueError("No conditional samples found for CFG residual loss")

    rand_ids = torch.randint(
        low=0,
        high=cond_indices.numel(),
        size=(iters, pair_batch_size),
        device=device,
    )
    cond = cond_indices[rand_ids]
    uncond = cond + n_samples
    return torch.cat([cond, uncond], dim=1)


def clone_index_state(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, list):
        return [clone_index_state(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clone_index_state(v) for v in obj)
    return obj


def move_index_state(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, list):
        return [move_index_state(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_index_state(v, device) for v in obj)
    return obj


def restore_quant_params(model, quant_params_dict):
    current = model.get_quant_params_dict()
    for name, value in quant_params_dict.items():
        if isinstance(value, list):
            for saved_dict, current_dict in zip(value, current[name]):
                for key, tensor in saved_dict.items():
                    if tensor is not None:
                        current_dict[key].data.copy_(tensor.to(current_dict[key].device, dtype=current_dict[key].dtype))
                    else:
                        current_dict[key] = None
        else:
            current[name].data.copy_(value.to(current[name].device, dtype=current[name].dtype))


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

    if opt_target == 'weight_and_activation':
        iters = config.quant.weight.optimization.iters
        assert config.quant.weight.optimization.iters == config.quant.activation.optimization.iters
    else:
        iters = getattr(config.quant,opt_target).optimization.iters
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
    loss_func = LossFunction(block, **config_loss)
    cfg_loss_weight = float(config_loss.get('cfg_loss_weight', 0.))
    save_interval = int(getattr(config.quant.weight.optimization, 'save_interval', 0))
    save_dir = getattr(config.quant.weight.optimization, 'save_dir', None)
    resume_ckpt = getattr(config.quant.weight.optimization, 'resume_ckpt', None)
    if save_interval > 0:
        if save_dir is None:
            logger.warning("save_interval is set but save_dir is missing; intermediate checkpoints will be skipped")
        else:
            os.makedirs(save_dir, exist_ok=True)

    # move to gpu device
    # sample_idxs = torch.randint(low=0,high=cached_inps.shape[0],size=(iters,batch_size))
    if isinstance(cached_inps, list):
        if isinstance(cached_outs, list):
            idxs_list = []
            for i in range(len(cached_outs)):
                idxs_list.append(torch.randint(low=0,high=cached_inps[1][i].shape[0],size=(iters,1), device=cached_inps[1][i].device))
            pmp_idxs = torch.randint(low=0,high=len(cached_outs),size=(iters, 1), device=cached_inps[0][0].device)
        else:
            if cfg_loss_weight > 0:
                pair_batch_size = max(1, batch_size // 2)
                sample_idxs = sample_cfg_pair_indices(
                    cached_inps[0].shape[0],
                    config.calib_data.n_samples,
                    iters,
                    pair_batch_size,
                    cached_inps[0].device,
                )
            else:
                sample_idxs = torch.randint(low=0,high=cached_inps[0].shape[0],size=(iters,batch_size), device=cached_inps[0].device)
    else:
        if cfg_loss_weight > 0:
            pair_batch_size = max(1, batch_size // 2)
            sample_idxs = sample_cfg_pair_indices(
                cached_inps.shape[0],
                config.calib_data.n_samples,
                iters,
                pair_batch_size,
                cached_inps.device,
            )
        else:
            sample_idxs = torch.randint(low=0,high=cached_inps.shape[0],size=(iters,batch_size), device=cached_inps.device)
    torch.set_grad_enabled(True)
    # import ipdb; ipdb.set_trace()
    # iters = 16 # debug
    if enable_fp32:
        scaler = GradScaler()
    start_iter = 0
    resume_state = None
    if resume_ckpt is not None:
        resume_state = torch.load(resume_ckpt, map_location="cpu")
        if resume_state.get("opt_target") != opt_target:
            raise ValueError(f"resume opt_target mismatch: {resume_state.get('opt_target')} vs {opt_target}")
        if int(resume_state.get("iters", iters)) != iters:
            raise ValueError(f"resume iters mismatch: {resume_state.get('iters')} vs {iters}")
        restore_quant_params(model, resume_state["quant_params"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        if enable_fp32 and "scaler" in resume_state and resume_state["scaler"] is not None:
            scaler.load_state_dict(resume_state["scaler"])
        loss_func.count = int(resume_state.get("loss_count", 0))
        start_iter = int(resume_state["iter"])
        if resume_state.get("sample_idxs") is not None:
            sample_idxs = move_index_state(resume_state["sample_idxs"], sample_idxs.device)
        if resume_state.get("idxs_list") is not None:
            idxs_list = move_index_state(resume_state["idxs_list"], idxs_list[0].device)
        if resume_state.get("pmp_idxs") is not None:
            pmp_idxs = move_index_state(resume_state["pmp_idxs"], pmp_idxs.device)
        if "torch_rng_state" in resume_state:
            torch.set_rng_state(resume_state["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in resume_state and resume_state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(resume_state["cuda_rng_state"], device=device)
        logger.info(f"Resuming reconstruction from {resume_ckpt} at iter {start_iter}/{iters}")

    for name, param in block.named_parameters():
        if ('lora' in name and 'minus' not in name) or 'delta' in name or 'mask' in name:
        # if 'lora' in name or 'zero_point' in name or 'delta' in name or 'zp_list' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # for name, param in block.named_parameters():
        # print(f"Parameter {name} requires_grad: {param.requires_grad}")

    for i in range(1, 27):
        set_grad_checkpoint(block.blocks[i])

    for i in range(start_iter, iters):
        # print(i)
        # import time
        # t0 = time.time()
        # idx = torch.randperm(cached_inps.size(0))[:batch_size]
        if isinstance(cached_outs, list):
            pmp_id = pmp_idxs[i]
            idx = idxs_list[pmp_id][i]
        else:
            idx = sample_idxs[i,:]
        # import ipdb; ipdb.set_trace()
        if isinstance(cached_inps, list):
            # 这个对应多输入
            if len(cached_inps)==2:
                # idx = torch.randperm(cached_inps[0].size(0))[:batch_size]
                cur_x = index_to_device(cached_inps[0], idx, device)
                cur_t = index_to_device(cached_inps[1], idx, device)
                cur_inp = (cur_x, cur_t)
            elif len(cached_inps)==3:
                # idx = torch.randperm(cached_inps[0].size(0))[:batch_size]
                cur_x = index_to_device(cached_inps[0], idx, device)
                cur_t = index_to_device(cached_inps[1], idx, device)
                cur_y = index_to_device(cached_inps[2], idx, device)
                cur_inp = (cur_x, cur_t, cur_y)
            else:
                # 针对 QuantTransformerBlock
                cur_inp = []
                # idx = torch.randperm(cached_inps[0].size(0))[:batch_size]
                for j in range(len(cached_inps)):
                    if j in [1]:
                        cur_inp.append(move_to_device(cached_inps[j][pmp_id][idx], device).requires_grad_())
                    elif j in [4]:
                        # 4 prob is None
                        if cached_inps[4] is None:
                            cur_inp.append(None)
                        else:
                            cur_inp.append(move_to_device(cached_inps[j][pmp_id][idx], device))
                    elif j in [3]:
                        cur_inp.append(move_to_device(cached_inps[j][pmp_id][idx], device))
                    else:
                        cur_inp.append(torch.cat([move_to_device(cached_inps[j][pmp_id][index], device) for index in [idx*4, idx*4+1, idx*4+2, idx*4+3]]).requires_grad_())
                    '''if cached_inps[j] == None:
                        cur_inp.append(None)
                    else:
                        cur_inp.append(cached_inps[j][idx])'''
                
                cur_inp = tuple(cur_inp)
        else:
            # idx = torch.randperm(cached_inps.size(0))[:batch_size]  # 随机取样
            cur_inp = index_to_device(cached_inps, idx, device)
        if isinstance(cached_outs, list):
            # cur_out = cached_outs[pmp_id][idx]
            cur_out = torch.cat([move_to_device(cached_outs[pmp_id][index], device) for index in [idx*4, idx*4+1, idx*4+2, idx*4+3]])
        else:
            cur_out = index_to_device(cached_outs, idx, device)
        cur_grad = cached_grads[idx] if use_grad else None

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
        
        if torch.isnan(err):
            import ipdb; ipdb.set_trace()
        if enable_fp32:
            scaler.scale(err).backward()
        else:
            err.backward()  # DEBUG_ONLY: cancel retrain_graph
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
        if scheduler:
            scheduler.step()
        if save_interval > 0 and save_dir is not None and (i + 1) % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"ckpt_iter{i + 1}.pth")
            torch.save(model.get_quant_params_dict(), ckpt_path)
            resume_path = os.path.join(save_dir, f"resume_iter{i + 1}.pth")
            torch.save({
                "iter": i + 1,
                "iters": iters,
                "opt_target": opt_target,
                "quant_params": model.get_quant_params_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if enable_fp32 else None,
                "loss_count": loss_func.count,
                "sample_idxs": clone_index_state(sample_idxs) if 'sample_idxs' in locals() else None,
                "idxs_list": clone_index_state(idxs_list) if 'idxs_list' in locals() else None,
                "pmp_idxs": clone_index_state(pmp_idxs) if 'pmp_idxs' in locals() else None,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device=device) if torch.cuda.is_available() else None,
            }, resume_path)
            logger.info(f"Saved intermediate quant checkpoint to {ckpt_path}")
            logger.info(f"Saved reconstruction resume checkpoint to {resume_path}")

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
