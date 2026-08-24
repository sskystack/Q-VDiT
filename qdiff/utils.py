import logging
import hashlib
import math
import time
from typing import Union
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from qdiff.models.quant_layer import QuantLayer
from qdiff.models.quant_block import BaseQuantBlock
from qdiff.models.quant_model import QuantModel
from qdiff.quantizer.base_quantizer import BaseQuantizer, lp_loss
from qdiff.mtd import (
    motion_transport_distillation,
    motion_transport_distillation_v2,
    normalize_mtd_config,
    normalize_mtd_v2_config,
)

logger = logging.getLogger(__name__)


class SemanticMTDCollector:
    """Collect reduced CFG and temporal-attention maps from one paired FP pass."""

    def __init__(self, model, attention_chunk_size):
        stdit = getattr(model, "model", None)
        if stdit is None or not hasattr(stdit, "blocks"):
            raise ValueError("semantic MTD collection requires an OpenSora STDiT model")
        if getattr(stdit, "enable_sequence_parallelism", False):
            raise ValueError("semantic MTD collection does not support sequence parallelism")
        self.stdit = stdit
        self.attention_chunk_size = int(attention_chunk_size)
        self.pair_count = 0
        self.handles = []
        self.temporal_by_block = {}
        self.cfg_sum = None
        self.temporal_sum = None
        self.block_count = 0

    def _spatial_map(self, values):
        spatial_tokens = values.shape[-1]
        side = math.isqrt(spatial_tokens)
        if side * side != spatial_tokens:
            raise ValueError(
                f"semantic MTD expected square spatial tokens, got {spatial_tokens}"
            )
        return values.reshape(self.pair_count, -1, side, side)

    def _attention_callback(self, block_index, incoming):
        total_batch = 2 * self.pair_count
        spatial_tokens = incoming.shape[0] // total_batch
        if incoming.shape[0] != total_batch * spatial_tokens:
            raise ValueError("temporal attention batch cannot be split into CFG pairs")
        temporal = incoming.reshape(
            total_batch, spatial_tokens, incoming.shape[-1]
        ).permute(0, 2, 1)
        self.temporal_by_block[block_index] = self._spatial_map(
            temporal[: self.pair_count].detach().float()
        )

    def _block_hook(self, block_index, module, inputs, output):
        del module, inputs
        if block_index not in self.temporal_by_block:
            raise RuntimeError(
                f"missing temporal attention statistics for STDiT block {block_index}"
            )
        if not torch.is_tensor(output) or output.shape[0] != 2 * self.pair_count:
            raise ValueError("STDiT block output does not match paired CFG layout")
        conditional = output[: self.pair_count].detach()
        unconditional = output[self.pair_count :].detach()
        cfg_chunks = []
        for start in range(0, output.shape[1], 1024):
            difference = (
                conditional[:, start : start + 1024].float()
                - unconditional[:, start : start + 1024].float()
            )
            cfg_chunks.append(difference.square().mean(dim=-1).sqrt())
        cfg = torch.cat(cfg_chunks, dim=1)
        cfg = cfg.reshape(self.pair_count, self.stdit.num_temporal, -1)
        cfg = self._spatial_map(cfg)
        temporal = self.temporal_by_block.pop(block_index)
        self.cfg_sum = cfg if self.cfg_sum is None else self.cfg_sum + cfg
        self.temporal_sum = (
            temporal if self.temporal_sum is None else self.temporal_sum + temporal
        )
        self.block_count += 1

    def start(self, pair_count):
        self.pair_count = int(pair_count)
        self.temporal_by_block = {}
        self.cfg_sum = None
        self.temporal_sum = None
        self.block_count = 0
        self.handles = []
        for index, block in enumerate(self.stdit.blocks):
            block.attn_temp.set_incoming_attention_callback(
                lambda incoming, index=index: self._attention_callback(index, incoming),
                chunk_size=self.attention_chunk_size,
            )
            self.handles.append(
                block.register_forward_hook(
                    lambda module, inputs, output, index=index: self._block_hook(
                        index, module, inputs, output
                    )
                )
            )

    def finish(self):
        for handle in self.handles:
            handle.remove()
        for block in self.stdit.blocks:
            block.attn_temp.set_incoming_attention_callback(None)
        self.handles = []
        if self.temporal_by_block:
            raise RuntimeError("unconsumed temporal attention statistics remain")
        if self.block_count != len(self.stdit.blocks):
            raise RuntimeError(
                f"collected {self.block_count} STDiT blocks, expected "
                f"{len(self.stdit.blocks)}"
            )
        return (
            self.cfg_sum.div(self.block_count).cpu(),
            self.temporal_sum.div(self.block_count).cpu(),
        )

    def abort(self):
        for handle in self.handles:
            handle.remove()
        for block in self.stdit.blocks:
            block.attn_temp.set_incoming_attention_callback(None)
        self.handles = []


def _robust_unit_interval(values, low_percentile, high_percentile, eps):
    flat = values.float().flatten()
    low = torch.quantile(flat, low_percentile / 100.0)
    high = torch.quantile(flat, high_percentile / 100.0)
    normalized = ((values.float() - low) / (high - low + eps)).clamp(0.0, 1.0)
    return normalized, float(low), float(high)


def _interleaved_cfg_pair_indices(step_start, pair_start, pair_count):
    """Return cond/uncond indices for [cond_0, uncond_0, ...] cache rows."""
    pair_ids = torch.arange(pair_start, pair_start + pair_count, dtype=torch.long)
    cond_indices = int(step_start) + 2 * pair_ids
    return cond_indices, cond_indices + 1

def get_quant_calib_data(config, sample_data, custom_steps=None, model_type='opensora', repeat_interleave=False):
    num_samples, num_st = config.calib_data.n_samples, custom_steps
    nsteps = len(sample_data["ts"])
    assert(nsteps >= custom_steps)  # custom_steps subsample the calib data
    if len(sample_data["ts"][0].shape) == 0:  # expand_dim for 0-dim tensor
        for i in range(nsteps):
            sample_data["ts"][i] = sample_data["ts"][i][None]

    # INFO: preprocess the batch-dim for CFG
    # sample_data has [2(cond & uncond), bs] layout
    # however, the ptq and quant_infer, we use batch size to index them
    # we need to permute it back into [bs,2] for batch choice (for QNN infer in PTQ)

    # for key in sample_data:
        # # shift back the dimension
        # shape_ = list(sample_data[key].shape)
        # sample_data[key] = sample_data[key].reshape([shape_[0]]+[2,shape_[1]//2]+shape_[2:])
        # sample_data[key] = sample_data[key].permute([0,2,1,*range(3,len(shape_)+1)])
        # sample_data[key] = sample_data[key].reshape(shape_)

    if not repeat_interleave:
        timesteps = list(range(0, nsteps, nsteps//num_st))
        logger.info(f'Selected {len(timesteps)} steps from {nsteps} sampling steps')

        xs_lst = [sample_data["xs"][i][:num_samples*2] for i in timesteps]
        ts_lst = [sample_data["ts"][i][:num_samples*2] for i in timesteps]
        cond_emb_lst = [sample_data["cond_emb"][i][:num_samples*2] for i in timesteps]
        mask_lst = [sample_data["mask"][i][:num_samples*2] for i in timesteps]
    else:
        ts_downsample_rate = nsteps // num_steps_chosen
        logger.info(f'Selected {len(timesteps)} steps from {nsteps} sampling steps')

        # INFO: classifier free guidance, have 2x for each sample
        xs_lst = [sample_data["xs"][i][:num_samples*2] for i in range(nsteps)]
        ts_lst = [sample_data["ts"][i][:num_samples*2] for i in timesteps]
        cond_emb_lst = [sample_data["cond_emb"][i][:num_samples*2] for i in range(nsteps)]
        mask_lst = [sample_data["mask"][i][:num_samples*2] for i in range(nsteps)]

    xs = torch.cat(xs_lst, dim=0)
    ts = torch.cat(ts_lst, dim=0)
    cond_embs = torch.cat(cond_emb_lst, dim=0)
    masks = torch.cat(mask_lst, dim=0)

    if model_type == 'opensora' or model_type == 'pixart':
        return xs, ts, cond_embs, masks
    else:
        raise NotImplementedError
    
@torch.no_grad()
def load_quant_params(qnn, ckpt_path, dtype=torch.float32):
    print("Loading quantized model checkpoint")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    qnn.set_module_name_for_quantizer(module=qnn.model)
    qnn.set_quant_params_dict(ckpt, dtype=dtype)

class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch 
        if self.store_output:
            self.output_store = output_batch 
        if self.stop_forward:
            import ipdb; ipdb.set_trace()
            raise StopForwardException

def pair_wise_sim_map_speed(feat):
    fea_0 = feat
    fea_1 = feat.transpose(1, 2)
    
    sim_map = torch.bmm(fea_0, fea_1)
    return sim_map.reshape(-1, sim_map.shape[-1])

def get_time_relation_loss(pred, tgt):
    # import ipdb; ipdb.set_trace()
    '''batch, b, n_frame, h, w = pred.shape
    pred = pred.reshape(batch*b, n_frame, -1)
    tgt = tgt.reshape(batch*b, n_frame, -1)
    s_sim_map = pair_wise_sim_map_speed(pred)
    t_sim_map = pair_wise_sim_map_speed(tgt)
    p_s = F.log_softmax(s_sim_map / 1.0, dim=1)
    p_t = F.softmax(t_sim_map / 1.0, dim=1)

    sim_dis = F.kl_div(p_s, p_t, reduction='mean')
    return sim_dis'''
    b, c, t, h, w = pred.shape
    pred = pred.reshape(b, t, -1)
    tgt = tgt.reshape(b, t, -1)
    pred = torch.nn.functional.normalize(pred, p=2, dim=2)
    tgt = torch.nn.functional.normalize(tgt, p=2, dim=2)
    s_sim_map = pair_wise_sim_map_speed(pred)
    t_sim_map = pair_wise_sim_map_speed(tgt)
    p_s = F.log_softmax(s_sim_map / 1.0, dim=1)
    p_t = F.softmax(t_sim_map / 1.0, dim=1)

    sim_dis = F.kl_div(p_s, p_t, reduction='mean')
    return sim_dis



class LossFunction:
    '''Wrapper of LossFunc, Get the round_loss and reconstruction_loss'''
    def __init__(self,
                 module,
                 round_loss_type: str = 'relaxation',
                 reconstruction_loss_type: str = 'mse',
                 lambda_coeff: float = 1.,  # the coeff between two loss
                 iters: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.,
                 module_type='layer',
                 use_reconstruction_loss=False,
                 use_round_loss=False,
                 mtd_config=None,
                 mtd_v2_schedule=None,
                 ):

        self.module = module
        self.module_type = module_type
        self.round_loss_type = round_loss_type
        self.reconstruction_loss_type = reconstruction_loss_type
        self.lambda_coeff = lambda_coeff
        self.loss_start = iters * warmup
        self.warmup = warmup
        self.iters = iters
        self.p = p
        self.use_reconstruction_loss = use_reconstruction_loss
        self.use_round_loss = use_round_loss
        self.mtd_config = normalize_mtd_config(mtd_config)
        self.mtd_v2_config = normalize_mtd_v2_config(mtd_config)
        self.mtd_v2_schedule = mtd_v2_schedule

        self.temp_decay = LinearTempDecay(iters, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(
        self,
        pred,
        tgt,
        grad=None,
        mtd_importance=None,
        mtd_fine_scores=None,
        mtd_v2_context=None,
    ):
        """
        Compute the total loss for adaptive rounding:
        reconstruction_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        # FlashAttention runs the transformer in BF16/FP16, but reconstruction
        # reductions are substantially more stable in FP32.  These casts are
        # placed only at the terminal loss boundary, so backward automatically
        # casts gradients back to the transformer's compute dtype before they
        # reach the FlashAttention kernel.
        pred = pred.float()
        tgt = tgt.float()
        if grad is not None:
            grad = grad.float()

        total_loss = 0.
        reconstruction_base = pred.new_zeros(())
        relation_loss_time = pred.new_zeros(())
        relation_loss_weighted = pred.new_zeros(())

        self.count += 1
        if self.use_reconstruction_loss:
            if self.reconstruction_loss_type == 'mse':
                reconstruction_base = lp_loss(pred, tgt, p=int(self.p), reduction='all')
                reconstruction_loss = reconstruction_base
            elif self.reconstruction_loss_type == 'relation':
                reconstruction_base = lp_loss(pred, tgt, p=int(self.p), reduction='all')
                relation_loss_time =  get_time_relation_loss(pred, tgt)
                relation_loss_weighted = relation_loss_time * 100.0
                reconstruction_loss = reconstruction_base + relation_loss_weighted
            elif self.reconstruction_loss_type == 'fisher_diag':
                reconstruction_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
            elif self.reconstruction_loss_type == 'fisher_full':
                a = (pred - tgt).abs()
                grad = grad.abs()
                batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
                reconstruction_loss = (batch_dotprod * a * grad).mean() / 100
            else:
                raise ValueError('Not supported reconstruction loss function: {}'.format(self.reconstruction_loss_type))
        else:
            reconstruction_loss = 0.

        call_mtd_config = dict(self.mtd_config)
        call_mtd_config["iteration"] = self.count
        mtd_loss, mtd_terms, mtd_diagnostics = motion_transport_distillation(
            pred,
            tgt,
            call_mtd_config,
            return_components=True,
            return_diagnostics=True,
            importance_weights=mtd_importance,
            fine_selection_scores=mtd_fine_scores,
        )
        if mtd_v2_context is not None:
            mtd_v2_context = dict(mtd_v2_context)
            mtd_v2_context["schedule"] = self.mtd_v2_schedule
        mtd_v2_loss, mtd_v2_terms, mtd_v2_diagnostics = (
            motion_transport_distillation_v2(
                pred,
                tgt,
                context=mtd_v2_context,
                config=self.mtd_v2_config,
                return_components=True,
                return_diagnostics=True,
            )
        )

        b = self.temp_decay(self.count)
        if self.use_round_loss:
            if self.count < self.loss_start or self.round_loss_type == 'none':
                b = round_loss = 0
            elif self.round_loss_type == 'relaxation':
                round_loss = 0
                # DEBUG: didnot consider split rounding error
                if self.module_type == 'layer':
                    round_vals = self.module.weight_quantizer.get_soft_targets()
                    round_loss += self.lambda_coeff * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
                elif self.module_type == 'block':
                    round_loss = 0
                    for name, module_ in self.module.named_modules():
                        if isinstance(module_, QuantLayer):
                            round_vals = module_.weight_quantizer.get_soft_targets()
                            round_loss += self.lambda_coeff * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
            else:
                raise NotImplementedError
        else:
            round_loss = 0.

        total_loss += reconstruction_loss
        total_loss += round_loss
        total_loss += mtd_loss
        total_loss += mtd_v2_loss
        # Keep graph-connected component tensors available to diagnostics in
        # the reconstruction loop.  They are cleared immediately after the
        # backward pass there, so this does not retain graphs across steps.
        self.last_component_tensors = {
            'reconstruction': reconstruction_loss,
            'reconstruction_base': reconstruction_base,
            'official_tmd_raw': relation_loss_time,
            'official_tmd_weighted': relation_loss_weighted,
            'mtd': mtd_loss,
            'mtd_local': mtd_terms['local'],
            'mtd_motion': mtd_terms['motion'],
            'mtd_global': mtd_terms['global'],
            'mtd_fine': mtd_terms['fine'],
            'mtd_v2': mtd_v2_loss,
            'mtd_v2_correspondence': mtd_v2_terms['correspondence'],
            'mtd_v2_flow': mtd_v2_terms['flow'],
            'mtd_v2_global': mtd_v2_terms['global'],
            'round': round_loss,
            'total': total_loss,
        }
        self.last_components = {
            'reconstruction': float(reconstruction_loss),
            'reconstruction_base': float(reconstruction_base),
            'official_tmd_raw': float(relation_loss_time),
            'official_tmd_weighted': float(relation_loss_weighted),
            'mtd': float(mtd_loss),
            'mtd_local': float(mtd_terms['local']),
            'mtd_motion': float(mtd_terms['motion']),
            'mtd_global': float(mtd_terms['global']),
            'mtd_fine': float(mtd_terms['fine']),
            'mtd_v2': float(mtd_v2_loss),
            'mtd_v2_correspondence': float(mtd_v2_terms['correspondence']),
            'mtd_v2_flow': float(mtd_v2_terms['flow']),
            'mtd_v2_global': float(mtd_v2_terms['global']),
            'round': float(round_loss),
            'total': float(total_loss),
        }
        self.last_mtd_diagnostics = {}
        for name, value in mtd_diagnostics.items():
            if torch.is_tensor(value):
                detached = value.detach().cpu()
                converted = (
                    float(detached) if detached.numel() == 1
                    else detached.tolist()
                )
            else:
                converted = value
            self.last_mtd_diagnostics[name] = converted
            self.last_components[f'mtd_{name}'] = converted
        for name, value in mtd_v2_diagnostics.items():
            if torch.is_tensor(value):
                detached = value.detach().cpu()
                converted = (
                    float(detached) if detached.numel() == 1
                    else detached.tolist()
                )
            else:
                converted = value
            self.last_mtd_diagnostics[f'v2_{name}'] = converted
            self.last_components[f'mtd_v2_{name}'] = converted
        if self.count == 1 or self.count % 100 == 0:
            reconstruction_loss = -1 if not self.use_reconstruction_loss else reconstruction_loss
            round_loss = -1 if not self.use_round_loss else round_loss
            logger.info(
                'Total loss:\t{:.6f} '
                '(rec:{:.6f}, mtd:{:.6f} [local:{:.6f}, motion:{:.6f}, '
                'global:{:.6f}, fine:{:.6f}], mtd_v2:{:.6f} '
                '[corr:{:.6f}, flow:{:.6f}, global:{:.6f}], round:{:.6})'
                '\tb={:.2f}\tcount={}'.format(
                    float(total_loss), float(reconstruction_loss), float(mtd_loss),
                    float(mtd_terms['local']), float(mtd_terms['motion']),
                    float(mtd_terms['global']),
                    float(mtd_terms['fine']),
                    float(mtd_v2_loss), float(mtd_v2_terms['correspondence']),
                    float(mtd_v2_terms['flow']), float(mtd_v2_terms['global']),
                    float(round_loss), b, self.count
                )
            )
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))

def prepare_coco_text_and_image(json_file):
    info = json.load(open(json_file, 'r'))
    annotation_list = info["annotations"]
    image_caption_dict = {}
    for annotation_dict in annotation_list:
        if annotation_dict["image_id"] in image_caption_dict.keys():
            image_caption_dict[annotation_dict["image_id"]].append(annotation_dict["caption"])
        else:
            image_caption_dict[annotation_dict["image_id"]] = [annotation_dict["caption"]]
    captions = list(image_caption_dict.values())
    image_ids = list(image_caption_dict.keys())

    active_captions = []
    for texts in captions:
        active_captions.append(texts[0])

    image_paths = []
    for image_id in image_ids:
        image_paths.append("/share/public/diffusion_quant/coco/coco/val2014/"+f"COCO_val2014_{image_id:012}.jpg")
    return active_captions, image_paths

def _save_semantic_reconstruction_cache(
    model,
    layer,
    calib_data,
    config,
    cache_device,
    mtd_config,
):
    if layer is not model or config.model.model_type != "opensora":
        raise ValueError("semantic MTD cache collection requires whole-model OpenSora reconstruction")
    calib_xs, calib_ts, calib_conds, calib_masks = calib_data
    samples_per_step = int(config.calib_data.n_samples)
    group_size = 2 * samples_per_step
    total_items = int(calib_xs.shape[0])
    if total_items % group_size:
        raise ValueError(
            "semantic MTD calibration data must contain cond/uncond groups of "
            f"{group_size}, got {total_items} items"
        )

    collector = SemanticMTDCollector(
        model,
        attention_chunk_size=mtd_config["attention_chunk_size"],
    )
    get_in_out = GetLayerInOut(
        model,
        layer,
        model_type="opensora",
        previous_layer_quantized=True,
        semantic_collector=collector,
    )
    cached_outs = None
    cached_xs = torch.empty_like(calib_xs)
    cached_ts = torch.empty_like(calib_ts)
    raw_cfg = None
    raw_temporal = None
    pair_batch_size = mtd_config["semantic_pair_batch_size"]
    start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=next(model.parameters()).device)

    for step_start in trange(0, total_items, group_size):
        for pair_start in range(0, samples_per_step, pair_batch_size):
            pair_count = min(pair_batch_size, samples_per_step - pair_start)
            cond_indices, uncond_indices = _interleaved_cfg_pair_indices(
                step_start, pair_start, pair_count
            )
            if not torch.equal(
                calib_ts.index_select(0, cond_indices),
                calib_ts.index_select(0, uncond_indices),
            ):
                raise ValueError("semantic MTD CFG pair timesteps do not match")

            # The calibration artifact stores an independent second latent
            # half, but CFG inference duplicates the first half before the
            # conditional/unconditional forwards.  Use that same effective
            # input here so the hidden-state difference isolates text.
            paired_x = calib_xs.index_select(0, cond_indices)
            paired_t = calib_ts.index_select(0, cond_indices)
            indices = torch.cat([cond_indices, uncond_indices])
            paired_x = torch.cat([paired_x, paired_x])
            paired_t = torch.cat([paired_t, paired_t])
            device = next(model.parameters()).device
            _, cur_out = get_in_out(
                paired_x.to(device),
                paired_t.to(device),
                calib_conds.index_select(0, indices).to(device),
                {},
                calib_masks.index_select(0, indices).to(device),
                semantic_pair_count=pair_count,
            )
            cfg_map, temporal_map = get_in_out.semantic_store
            if cached_outs is None:
                cached_outs = torch.empty(
                    total_items, *cur_out.shape[1:], dtype=cur_out.dtype
                )
                raw_cfg = torch.empty(
                    total_items, *cfg_map.shape[1:], dtype=torch.float32
                )
                raw_temporal = torch.empty_like(raw_cfg)
            cached_outs.index_copy_(0, indices, cur_out.detach().cpu())
            cached_xs.index_copy_(0, indices, paired_x)
            cached_ts.index_copy_(0, indices, paired_t)
            raw_cfg.index_copy_(0, cond_indices, cfg_map)
            raw_cfg.index_copy_(0, uncond_indices, cfg_map)
            raw_temporal.index_copy_(0, cond_indices, temporal_map)
            raw_temporal.index_copy_(0, uncond_indices, temporal_map)

    low = mtd_config["semantic_percentile_low"]
    high = mtd_config["semantic_percentile_high"]
    eps = mtd_config["importance_eps"]
    normalized_cfg, cfg_low, cfg_high = _robust_unit_interval(
        raw_cfg, low, high, eps
    )
    normalized_temporal, temporal_low, temporal_high = _robust_unit_interval(
        raw_temporal, low, high, eps
    )
    native_product = normalized_cfg * normalized_temporal

    def resize_scores(scores, size):
        batch, frames, height, width = scores.shape
        resized = F.adaptive_avg_pool2d(
            scores.reshape(batch * frames, 1, height, width),
            (size, size),
        )
        return resized.reshape(batch, frames, size, size)

    coarse_product = resize_scores(native_product, mtd_config["transport_size"])
    importance = (
        mtd_config["weight_min"]
        + (mtd_config["weight_max"] - mtd_config["weight_min"])
        * coarse_product
    )
    importance = importance[:, :-1].flatten(2).contiguous()
    fine_scores = resize_scores(
        native_product, mtd_config["fine_transport_size"]
    )[:, :-1].flatten(2).contiguous()
    importance_hash = hashlib.sha256(importance.numpy().tobytes()).hexdigest()
    fine_score_hash = hashlib.sha256(fine_scores.numpy().tobytes()).hexdigest()
    elapsed = time.perf_counter() - start_time
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device=next(model.parameters()).device))
        if torch.cuda.is_available()
        else 0
    )
    statistics = {
        "cfg_p_low": cfg_low,
        "cfg_p_high": cfg_high,
        "temporal_p_low": temporal_low,
        "temporal_p_high": temporal_high,
        "weight_min_observed": float(importance.min()),
        "weight_max_observed": float(importance.max()),
        "importance_hash": importance_hash,
        "fine_score_min_observed": float(fine_scores.min()),
        "fine_score_max_observed": float(fine_scores.max()),
        "fine_score_hash": fine_score_hash,
        "cfg_pair_layout": "interleaved_cond_uncond",
        "cfg_pair_latent_policy": "duplicate_conditional_latent",
    }
    logger.info(
        "Semantic MTD cache: time=%.2fs peak_memory=%d MiB "
        "cfg_p=(%.6g, %.6g) temporal_p=(%.6g, %.6g) weights=(%.4f, %.4f)",
        elapsed,
        peak_memory // (1024 * 1024),
        cfg_low,
        cfg_high,
        temporal_low,
        temporal_high,
        statistics["weight_min_observed"],
        statistics["weight_max_observed"],
    )
    cached_inps = [cached_xs, cached_ts, calib_conds]
    cached_inps = [value.to(cache_device) for value in cached_inps]
    return (
        cached_inps,
        cached_outs.to(cache_device),
        importance.to(cache_device),
        fine_scores.to(cache_device),
        statistics,
    )


# ---------- save input output activation & grad ---------------------
def save_in_out_data(model: QuantModel, layer: Union[QuantLayer, BaseQuantBlock], calib_data: torch.Tensor, config, model_type='sdxl', split_save_attn=False, collect_mtd_importance=False):
    # asym: bool = False, act_quant: bool = False, batch_size: int = 32, keep_gpu: bool = True,
                      # cond: bool = True, split_save_attn: bool = False, model_type='sdxl'):
    """
    Save input data and output data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantLayer or QuantBlock
    :param calib_data: calibration data set
    :param asym: if Ture, save quantized input and full precision output
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :param cond: conditional generation or not
    :param split_save_attn: avoid OOM when caching n^2 attention matrix when n is large
    :return: input and output data
    """
    device = next(model.parameters()).device
    cache_device_name = str(getattr(config, 'reconstruction_cache_device', 'cuda')).lower()
    if cache_device_name not in ('cpu', 'cuda'):
        raise ValueError(
            "reconstruction_cache_device must be either 'cpu' or 'cuda', "
            f"got {cache_device_name!r}"
        )
    cache_device = torch.device('cpu') if cache_device_name == 'cpu' else device
    mtd_config = normalize_mtd_config(config)
    if collect_mtd_importance:
        if not mtd_config["semantic_importance"]:
            raise ValueError("collect_mtd_importance requires semantic_importance")
        return _save_semantic_reconstruction_cache(
            model, layer, calib_data, config, cache_device, mtd_config
        )
    get_in_out = GetLayerInOut(model, layer, model_type=model_type, previous_layer_quantized=True)
    cached_batches = []
    cached_inps, cached_outs = None, None
    torch.cuda.empty_cache()

    assert config.conditional # only support conditional generation
    assert not split_save_attn, "not checked for now"

    if model_type == 'sdxl':
        calib_xs, calib_ts, calib_conds, calib_added_conds = calib_data
        calib_added_text_embeds = calib_added_conds["text_embeds"]
        calib_added_time_ids = calib_added_conds["time_ids"]
        calib_masks = None
    elif model_type == 'sd':
        calib_xs, calib_ts, calib_conds = calib_data
        calib_masks = None
    elif model_type == 'pixart' or model_type == 'opensora':
        calib_xs, calib_ts, calib_conds, calib_masks = calib_data
    else:
        raise NotImplementedError

    # INO: whether split attention map to avoid OOM
    # if split_save_attn:
    #     logger.info("Checking if attention is too large...")

    #     if model_type == 'sdxl':
    #         calib_added_conds["text_embeds"] = calib_added_text_embeds[:1].to(device)
    #         calib_added_conds["time_ids"] = calib_added_time_ids[:1].to(device)
    #     test_inp, test_out = get_in_out(
    #         calib_xs[:1].to(device),
    #         calib_ts[:1].to(device),
    #         calib_conds[:1].to(device),
    #         calib_added_conds,
    #     )

    #     split_save_attn = False
    #     if (isinstance(test_inp, tuple) and test_inp[0].shape[1] == test_inp[0].shape[2]):
    #         logger.info(f"test_inp shape: {test_inp[0].shape}, {test_inp[1].shape}")
    #         if test_inp[0].shape[1] == 4096:
    #             split_save_attn = True
    #     if test_out.shape[1] == test_out.shape[2]:
    #         logger.info(f"test_out shape: {test_out.shape}")
    #         if test_out.shape[1] == 4096:
    #             split_save_attn = True

    #     if split_save_attn:
    #         logger.info("Confirmed. Trading speed for memory when caching attn matrix calibration data")
    #         inds = np.random.choice(calib_xs.size(0), calib_xs.size(0) // 2, replace=False)
    #     else:
    #         logger.info("Nope. Using normal caching method")

    batch_size = config.calib_data.batch_size
    iters = int(calib_xs.size(0) / batch_size)
    l_in_0, l_in_1, l_in, l_out = 0, 0, 0, 0
    if split_save_attn:
        num //= 2

    # INFO: iter through all the calib_data, save all input and output
    # defaults
    calib_masks_ = None
    tmp_kwargs = {}
    calib_added_conds = {}
    # iters = 2  # DEBUG_ONLY: not using the whole calib data
    for i in trange(iters):
        if model_type == 'sdxl':
            calib_added_conds["text_embeds"] = calib_added_text_embeds[i * batch_size:(i + 1) * batch_size].to(device)
            calib_added_conds["time_ids"] = calib_added_time_ids[i * batch_size:(i + 1) * batch_size].to(device)
        elif model_type == 'pixart':
            calib_masks_ = calib_masks[i * batch_size:(i + 1) * batch_size].to(device)
            tmp_kwargs = {
                'fcf': 4.5,
                'data_info': {
                    'img_hw': torch.tensor([[1024., 1024.]], device=device),
                    'aspect_ratio': torch.tensor([[1.]], device=device)
                }
            }
        elif model_type == 'opensora':
            calib_masks_ = calib_masks[i * batch_size:(i + 1) * batch_size].to(device)
        else:
            pass

        cur_inp, cur_out = get_in_out(
            calib_xs[i * batch_size:(i + 1) * batch_size].to(device),
            calib_ts[i * batch_size:(i + 1) * batch_size].to(device),
            calib_conds[i * batch_size:(i + 1) * batch_size].to(device),
            calib_added_conds,
            calib_masks_,
        )
        # import ipdb; ipdb.set_trace()
        if isinstance(cur_inp, tuple):
            if(len(cur_inp)>2):
                cur_inp = list(cur_inp)
                for i in range(len(cur_inp)):
                    if isinstance(cur_inp[i], list):
                        pass
                    else:
                        cur_inp[i] = cur_inp[i].cpu() if cur_inp[i] is not None else None  # difference
                cached_batches.append((tuple(cur_inp), cur_out.cpu()))
            else:
                cur_x, cur_t = cur_inp
                if not split_save_attn:
                    cached_batches.append(((cur_x.cpu(), cur_t.cpu()), cur_out.cpu()))
                else:
                    if cached_inps is None:
                        l_in_0 = cur_x.shape[0] * num
                        l_in_1 = cur_t.shape[0] * num
                        cached_inps = [torch.zeros(l_in_0, *cur_x.shape[1:]), torch.zeros(l_in_1, *cur_t.shape[1:])]
                    cached_inps[0].index_copy_(0, torch.arange(i * cur_x.shape[0], (i + 1) * cur_x.shape[0]), cur_x.cpu())
                    cached_inps[1].index_copy_(0, torch.arange(i * cur_t.shape[0], (i + 1) * cur_t.shape[0]), cur_t.cpu())
        else:
            if not split_save_attn:
                cached_batches.append((cur_inp.cpu(), cur_out.cpu()))
            else:
                if cached_inps is None:
                    l_in = cur_inp.shape[0] * num
                    cached_inps = torch.zeros(l_in, *cur_inp.shape[1:])
                cached_inps.index_copy_(0, torch.arange(i * cur_inp.shape[0], (i + 1) * cur_inp.shape[0]), cur_inp.cpu())

        # if split_save_attn:
        #     if cached_outs is None:
        #         l_out = cur_out.shape[0] * num
        #         cached_outs = torch.zeros(l_out, *cur_out.shape[1:])
        #     cached_outs.index_copy_(0, torch.arange(i * cur_out.shape[0], (i + 1) * cur_out.shape[0]), cur_out.cpu())
    
    # cached_batches[0][0] len is 5, cached_batches[0][0][3] is list, cached_batches[0][0][1] shape different, but i think only 10 candidate
    # import ipdb; ipdb.set_trace()
    if not split_save_attn:
        if isinstance(cached_batches[0][0], tuple):
            # if input_type in tuple, QuantTransformerBlock
            if len(cached_batches[0][0]) > 3:
                shape_to_indices = {}
                cached_inps = []
                cached_outs = []
                # 遍历cached_batches
                for index, batch in enumerate(cached_batches):
                    # 提取shape
                    shape = batch[0][1].shape
                    if shape not in shape_to_indices:
                        shape_to_indices[shape] = []
                    shape_to_indices[shape].append(index)
                for i in range(len(cached_batches[0][0])):
                    tmp_list = []
                    if isinstance(cached_batches[0][0][i], list):
                        for shape, indices in shape_to_indices.items():
                            tmp_list.append(list(cached_batches[indice][0][i] for indice in indices))
                    else:
                        if cached_batches[0][0][i] is None:
                            tmp_list = None
                        else:
                            for shape, indices in shape_to_indices.items():
                                tmp_list.append(torch.cat([cached_batches[indice][0][i] for indice in indices]))
                    cached_inps.append(tmp_list)
                for shape, indices in shape_to_indices.items():
                    cached_outs.append(torch.cat([cached_batches[indice][1] for indice in indices]))
                
                # import ipdb; ipdb.set_trace()
                
                '''for i in range(len(cached_batches[0][0])):
                    if cached_batches[0][0][i] == None:
                        cached_inps.append(None)
                    else:
                        cached_inps.append(torch.cat([x[0][i] for x in cached_batches]))  # difference'''
            elif len(cached_batches[0][0]) == 3:
                cached_inps = [
                    torch.cat([x[0][0] for x in cached_batches]),
                    torch.cat([x[0][1] for x in cached_batches]),
                    torch.cat([x[0][2] for x in cached_batches])
                ]
                cached_outs = torch.cat([x[1] for x in cached_batches])
            else:
                cached_inps = [
                    torch.cat([x[0][0] for x in cached_batches]),
                    torch.cat([x[0][1] for x in cached_batches])
                ]
        else:
            cached_inps = torch.cat([x[0] for x in cached_batches])
            cached_outs = torch.cat([x[1] for x in cached_batches])

    if isinstance(cached_inps, list):
        if isinstance(cached_inps[0], list):
            pass
        else:
            for i in range(len(cached_inps)):
                logger.info(f"in {i} shape: {cached_inps[i].shape}") if cached_inps[i] is not None else logger.info(f"in {i} : None") 
    else:
        logger.info(f"in shape: {cached_inps.shape}")
    if isinstance(cached_outs, list):
        pass
    else:
        logger.info(f"out shape: {cached_outs.shape}")
    torch.cuda.empty_cache()

    # Keep the complete reconstruction cache on the configured device.  CPU
    # caching avoids holding several GiB of immutable calibration tensors on
    # the GPU; block_reconstruction moves only the selected mini-batch back to
    # the model device.  A recursive move is required for OpenSora's nested
    # per-shape cache, including list-valued attention metadata.
    def move_cache(value):
        if torch.is_tensor(value):
            return value.to(cache_device)
        if isinstance(value, list):
            return [move_cache(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move_cache(item) for item in value)
        if isinstance(value, dict):
            return {key: move_cache(item) for key, item in value.items()}
        return value

    cached_inps = move_cache(cached_inps)
    cached_outs = move_cache(cached_outs)
    if cache_device.type == 'cpu':
        torch.cuda.empty_cache()
    logger.info("Reconstruction cache device: %s", cache_device)

    return cached_inps, cached_outs

def save_grad_data(model: QuantModel, layer: Union[QuantLayer, BaseQuantBlock], calib_data: torch.Tensor,
                   damping: float = 1., act_quant: bool = False, batch_size: int = 32,
                   keep_gpu: bool = True):
    """
    Save gradient data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantLayer or QuantBlock
    :param calib_data: calibration data set
    :param damping: damping the second-order gradient by adding some constant in the FIM diagonal
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: gradient data
    """
    device = next(model.parameters()).device
    get_grad = GetLayerGrad(model, layer, device, act_quant=act_quant)
    cached_batches = []
    torch.cuda.empty_cache()

    for i in range(int(calib_data[0].size(0) / batch_size)):
        cur_grad = get_grad(calib_data[0][i * batch_size:(i + 1) * batch_size])
        cached_batches.append(cur_grad.cpu())

    cached_grads = torch.cat([x for x in cached_batches])
    cached_grads = cached_grads.abs() + 1.0
    # scaling to make sure its mean is 1
    # cached_grads = cached_grads * torch.sqrt(cached_grads.numel() / cached_grads.pow(2).sum())
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_grads = cached_grads.to(device)
    return cached_grads


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class GetLayerInOut:
    def __init__(self, model: QuantModel, layer: Union[QuantLayer, BaseQuantBlock], model_type='sd', previous_layer_quantized=False, semantic_collector=None):
                 # device: torch.device, asym: bool = False, act_quant: bool = False, model_type='sd'):
        self.model = model
        self.layer = layer
        self.previous_layer_quantized = previous_layer_quantized
        self.semantic_collector = semantic_collector
        self.semantic_store = None
        # self.device = device
        # self.act_quant = act_quant
        self.model_type = model_type
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, x, timesteps, context=None, added_conds=None, mask=None, semantic_pair_count=None):

        self.model.eval()  # temporarily use eval mode
        # INFO: save the quant_state, since it will be written by (False, False)
        model_quant_weight, model_quant_act = self.model.get_quant_state()
        if isinstance(self.layer, QuantLayer):
            layer_quant_weight, layer_quant_act = self.layer.get_quant_state()
        self.model.set_quant_state(False, False)  # use all FP model
        handle = self.layer.register_forward_hook(self.data_saver)

        with torch.no_grad():
            # for pixart
            if self.model_type == 'pixart':
                tmp_kwargs = {
                    'fcf': 4.5,
                    'data_info': {
                        'img_hw': torch.tensor([[1024., 1024.]], device='cuda'),
                        'aspect_ratio': torch.tensor([[1.]], device='cuda')
                    }
                }
            else:
                tmp_kwargs = {}

            if self.semantic_collector is not None:
                self.semantic_collector.start(semantic_pair_count)
            try:
                try:
                    if self.model_type == 'opensora':
                        _ = self.model(x, timesteps, context, mask=mask, **tmp_kwargs)
                    else:
                        _ = self.model(x, timesteps, context, added_cond_kwargs=added_conds, mask=mask, **tmp_kwargs)
                except StopForwardException:
                    pass
            finally:
                if self.semantic_collector is not None:
                    self.semantic_store = self.semantic_collector.finish()

            if self.previous_layer_quantized:
                # INFO: rewrite the input data, with *all previous layer* quantized
                # overwrite input with network quantized
                self.data_saver.store_output = False  # avoid overwrite the output data
                self.model.set_quant_state(model_quant_weight, model_quant_act)  # restore original model quant_state
                try:
                    if self.model_type == 'opensora':
                        _ = self.model(x, timesteps, context, mask=mask, **tmp_kwargs)
                    else:
                        _ = self.model(x, timesteps, context, added_cond_kwargs=added_conds, mask=mask, **tmp_kwargs)
                except StopForwardException:
                    pass
                self.data_saver.store_output = True
        handle.remove()

        self.model.set_quant_state(model_quant_weight, model_quant_act)
        if isinstance(self.layer, QuantLayer):
            self.layer.set_quant_state(layer_quant_weight, layer_quant_act)
        self.model.train()

        # import ipdb; ipdb.set_trace()
        if len(self.data_saver.input_store) > 1 and len(self.data_saver.input_store) < 3 and torch.is_tensor(self.data_saver.input_store[1]):
            return (self.data_saver.input_store[0].detach(),
                self.data_saver.input_store[1].detach()), self.data_saver.output_store.detach()
        elif len(self.data_saver.input_store) == 3:
            input_tuple = []
            # import ipdb; ipdb.set_trace()
            for input in self.data_saver.input_store:
                if input == None:
                    input_tuple.append(input)
                else:
                    if torch.is_tensor(input):
                        input_tuple.append(input.detach())
                    else:
                        input_tuple.append(input)
            return tuple(input_tuple), self.data_saver.output_store.detach()  # difference
        elif len(self.data_saver.input_store) == 5:
            input_tuple = []
            # import ipdb; ipdb.set_trace()
            for input in self.data_saver.input_store:
                if input == None:
                    input_tuple.append(input)
                else:
                    if torch.is_tensor(input):
                        input_tuple.append(input.detach())
                    else:
                        input_tuple.append(input)
            return tuple(input_tuple), self.data_saver.output_store.detach()  # difference
        elif len(self.data_saver.input_store) == 7:
            input_tuple = []
            for input in self.data_saver.input_store:
                if input == None:
                    input_tuple.append(input)
                else:
                    input_tuple.append(input.detach())
            return tuple(input_tuple), self.data_saver.output_store.detach()  # difference
        else:
            return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach()

class GradSaverHook:
    def __init__(self, store_grad=True):
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = None

    def __call__(self, module, grad_input, grad_output):
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException


class GetLayerGrad:
    def __init__(self, model: QuantModel, layer: Union[QuantLayer, BaseQuantBlock],
                 device: torch.device, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.device = device
        self.act_quant = act_quant
        self.data_saver = GradSaverHook(True)

    def __call__(self, model_input):
        """
        Compute the gradients of block output, note that we compute the
        gradient by calculating the KL loss between fp model and quant model

        :param model_input: calibration data samples
        :return: gradients
        """
        self.model.eval()

        handle = self.layer.register_backward_hook(self.data_saver)
        with torch.enable_grad():
            try:
                self.model.zero_grad()
                inputs = model_input.to(self.device)
                self.model.set_quant_state(False, False)
                out_fp = self.model(inputs)
                quantize_model_till(self.model, self.layer, self.act_quant)
                out_q = self.model(inputs)
                loss = F.kl_div(F.log_softmax(out_q, dim=1), F.softmax(out_fp, dim=1), reduction='batchmean')
                loss.backward()
            except StopForwardException:
                pass

        handle.remove()
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()
        return self.data_saver.grad_out.data


def quantize_model_till(model: QuantLayer, layer: Union[QuantLayer, BaseQuantBlock], act_quant: bool = False):
    """
    We assumes modules are correctly ordered, holds for all models considered
    :param model: quantized_model
    :param layer: a block or a single layer.
    """
    model.set_quant_state(False, False)
    for name, module in model.named_modules():
        if isinstance(module, (QuantLayer, BaseQuantBlock)):
            module.set_quant_state(True, act_quant)
        if module == layer:
            break
