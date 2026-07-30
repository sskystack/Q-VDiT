import os
import sys
import json
# sys.path.append(".")

import torch
import shutil
import logging
from omegaconf import OmegaConf
from mmengine.runner import set_random_seed
import yaml

from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.build_model import build_models
from opensora.utils.misc import to_torch_dtype

from qdiff.models.quant_model import QuantModel
from qdiff.utils import load_quant_params

import inspect
import os

def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts


def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False, mode="quant_inference")
    print(cfg)
    PRECOMPUTE_TEXT_EMBEDS = cfg.get('precompute_text_embeds', None)

    opt = cfg
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # INFO: add bakup file and bakup cfg into logpath for debug
    # load the config from the log path
    if not hasattr(opt,"calib_config"):
        opt.calib_config = os.path.join(opt.outdir,'config.yaml')
    if not hasattr(opt,"quant_ckpt") or not os.path.exists(opt.quant_ckpt):
        opt.quant_ckpt = os.path.join(opt.outdir,'ckpt.pth')
    if not hasattr(opt,"save_dir"):
        opt.save_dir = os.path.join(opt.outdir,'generated_videos')
    config = OmegaConf.load(f"{opt.calib_config}")

    log_path = os.path.join(outpath, "quant_inference_run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Conducting Command: %s", " ".join(sys.argv))

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    gpus = [int(d) for d in cfg.gpu.split(",")]
    torch.cuda.set_device(gpus[0])
    dtype = to_torch_dtype(cfg.dtype)
    print(dtype)
    
    set_random_seed(seed=cfg.seed)
    all_prompts = load_prompts(cfg.prompt_path)
    prompt_start_index = int(cfg.get("prompt_start_index", 0))
    requested_prompt_indices = cfg.get("prompt_indices", None)
    replay_original_prompt_rng = bool(cfg.get("replay_original_prompt_rng", False))
    if requested_prompt_indices:
        original_prompt_indices = [int(index) for index in requested_prompt_indices]
        invalid = [
            index for index in original_prompt_indices
            if not (0 <= index < len(all_prompts))
        ]
        if invalid:
            raise ValueError(
                f"prompt_indices contains values outside [0, {len(all_prompts)}): {invalid}"
            )
        if replay_original_prompt_rng:
            if int(cfg.batch_size) != 1:
                raise ValueError("replay_original_prompt_rng requires batch_size=1")
            if original_prompt_indices != sorted(original_prompt_indices):
                raise ValueError(
                    "replay_original_prompt_rng requires strictly increasing prompt_indices"
                )
            if len(set(original_prompt_indices)) != len(original_prompt_indices):
                raise ValueError(
                    "replay_original_prompt_rng does not allow duplicate prompt_indices"
                )
        prompts = [all_prompts[index] for index in original_prompt_indices]
    else:
        prompt_end_index = prompt_start_index + int(cfg.num_videos)
        if not (0 <= prompt_start_index < len(all_prompts)):
            raise ValueError(
                f"prompt_start_index={prompt_start_index} is outside prompt file "
                f"with {len(all_prompts)} entries"
            )
        original_prompt_indices = list(range(prompt_start_index, prompt_end_index))
        prompts = all_prompts[prompt_start_index:prompt_end_index]

    # ======================================================
    # 3. build model & load weights
    # ======================================================
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # 3.2. build model
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        # caption_channels=text_encoder.output_dim,
        caption_channels=4096,  # DIRTY: for T5 only
        model_max_length=cfg.text_encoder.model_max_length,
        dtype=dtype,
        enable_sequence_parallelism=False,
    )
    if PRECOMPUTE_TEXT_EMBEDS is not None:
        text_encoder = None
    else:
        text_encoder = build_module(cfg.text_encoder, MODELS, device=device)  # T5 must be fp32
        text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # 3.3. move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    # 3.4. support for multi-resolution
    model_args = dict()
    if cfg.multi_resolution:
        image_size = cfg.image_size
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        model_args["data_info"] = dict(ar=ar, hw=hw)


    # scheduler, model, text_encoder, vae, model_args, latent_size = build_models(cfg, device, dtype, enable_sequence_parallelism=False)
    # assert(config.conditional)
    # ======================================================
    # 4. get quantized model
    # ======================================================
    # Use the actual inference scheduler length. The calibration YAML may have
    # been collected with a different number of steps (for example 50-step
    # calibration followed by 100-step inference).
    num_sampling_timesteps = int(scheduler.num_timesteps)

    assert(config.conditional)

    wq_params = config.quant.weight.quantizer
    aq_params = config.quant.activation.quantizer
    use_weight_quant = True if wq_params else False
    use_act_quant = True if aq_params else False
    if opt.skip_quant_weight:
        use_weight_quant = False
    if opt.skip_quant_act:
        use_act_quant = False

    if config.get('mixed_precision', False):
        if use_weight_quant:
            wq_params['mixed_precision'] = config.mixed_precision
        # if use_act_quant:
        #     aq_params['mixed_precision'] = config.mixed_precision

    qnn = QuantModel(
        model=model, \
        weight_quant_params=wq_params,\
        act_quant_params=aq_params,\
        # act_quant_mode="qdiff",\
        # sm_abit=config.quant.softmax.n_bits,\
    )
    qnn.cuda()
    qnn.eval()
    logger.info(qnn)

    # DIRTY: set the cfg_split as the attribute of the model
    # the cfg_split is configured in `opensora/schedulers/ippdm/__init__.py`
    cfg_split = config.get('cfg_split', False)
    qnn.cfg_split = cfg_split


    qnn.set_quant_state(False, False)
    # for smooth quant
    qnn.set_smooth_quant(smooth_quant=False, smooth_quant_running_stat=False)
    calib_added_cond = {} # It is not required for STDiT

    # with torch.no_grad():
        # if "opensora" in config.model.model_id:
            # _ = qnn(torch.randn(1, 4, 16, 64, 64).cuda(), torch.randint(0, 1000, (1,)).cuda(), torch.randn(1, 1, 120, 4096).cuda(), mask=torch.ones(1, 120).cuda().to(torch.int64))
        # else:
            # raise NotImplementedError

    # for part quantization
    if opt.part_quant:
        quant_layer_list = list(torch.load(config.part_quant_list))
        quant_layer_list = quant_layer_list[:int(len(quant_layer_list) * opt.quant_ratio)]

    if opt.part_fp:
        with open(config.part_fp_list,'r') as f:
            lines = f.readlines()
        fp_layer_list = [line.strip() for line in lines]  # strip the '\n'
        if opt.get('fp_ratio',None) is not None:
            fp_layer_list = fp_layer_list[:int(len(fp_layer_list) * opt.fp_ratio)]
        logger.info("Set the following layers as FP: {}".format(fp_layer_list))

    # for smooth quant
    if aq_params.smooth_quant.enable:
        qnn.set_smooth_quant(smooth_quant=False, smooth_quant_running_stat=False)
        # for i in range(len(sens)):
        #     if sens[i][1]["fp16_diff"] > 7.0 or sens[i][1]["fp16_diff"] < 0.5:
        #         smooth_quant_layer_list.append(sens[i][0])
                # alpha_dict[sens[i][0]] = sens[i][1]["best_alpha"]
        # alpha_dict["model.blocks.27.mlp.fc2"] = 0.675
        qnn.set_smooth_quant(smooth_quant=True, smooth_quant_running_stat=False) # Now we use fp16 to save the statistic of activation
        qnn.set_layer_smooth_quant(model=qnn, module_name_list=fp_layer_list, smooth_quant=False, smooth_quant_running_stat=False)
        # qnn.set_layer_smooth_quant_alpha(model=qnn, alpha_dict=alpha_dict)

    # set the init flag True, otherwise will recalculate params
    if opt.part_quant:
        qnn.set_layer_quant(model=qnn, module_name_list=quant_layer_list, quant_level='per_layer', weight_quant=use_weight_quant, act_quant=use_act_quant, prefix="")
    elif opt.part_fp:
        qnn.set_quant_state(use_weight_quant, use_act_quant)
        qnn.set_layer_quant(model=qnn, module_name_list=fp_layer_list, quant_level='per_layer', weight_quant=False, act_quant=False, prefix="")
    else:
        qnn.set_quant_state(use_weight_quant, use_act_quant) # enable weight quantization, disable act quantization
    
    if wq_params.n_bits <= 4:
        with open(opt.time_mp_config_weight, 'r') as f:
            time_mp_config_weight = yaml.safe_load(f)
        qnn.load_bitwidth_config(model=qnn, bit_config=time_mp_config_weight, bit_type='weight')
    if aq_params.n_bits <= 6:
        with open(opt.time_mp_config_act, 'r') as f:
            time_mp_config_act = yaml.safe_load(f)
        qnn.load_bitwidth_config(model=qnn, bit_config=time_mp_config_act, bit_type='act')
    
    qnn.set_quant_init_done('weight')
    qnn.set_quant_init_done('activation')

    load_quant_params(qnn, opt.quant_ckpt)
    qnn.cuda()
    # The backbone was already moved to the requested inference dtype before
    # QuantModel wrapping.  Keep TQE master weights and static weight-quantizer
    # parameters in FP32, matching calib.py. QuantLayer casts the final
    # effective weight to the hidden-state dtype immediately before GEMM.
    # Casting the whole qnn to FP16 here can overflow optimized TQE/quantizer
    # intermediates even though the resulting linear output is representable.
    if dtype == torch.float32:
        qnn.to(dtype)

    # ======================================================
    # 5. inference
    # ======================================================
    qnn.use_weight_quant = use_weight_quant
    qnn.use_act_quant = use_act_quant
    qnn.layer_wise_quant = bool(opt.layer_wise_quant)
    qnn.group_wise_quant = bool(opt.group_wise_quant)
    qnn.block_group_wise_quant = bool(opt.block_group_wise_quant)
    qnn.timestep_wise_quant = bool(opt.timestep_wise_quant)
    qnn.timestep_fp_layer_list = (
        fp_layer_list
        if opt.part_fp
        else ["x_embedder", "t_block", "t_embedder", "y_embedder", "final_layer"]
    )
    qnn.mtd_profile_enabled = bool(opt.get("mtd_profile_dir", None))
    qnn.mtd_profile_dir = opt.get("mtd_profile_dir", None)
    qnn.mtd_profile_steps = set(
        int(step) for step in (opt.get("mtd_profile_steps", None) or [])
    )
    qnn.mtd_profile_transport_size = int(
        opt.get("mtd_profile_transport_size", 16)
    )
    if qnn.mtd_profile_enabled:
        if not qnn.mtd_profile_steps:
            raise ValueError("--mtd_profile_dir requires --mtd_profile_steps")
        invalid_steps = [
            step for step in qnn.mtd_profile_steps
            if not (1 <= step <= num_sampling_timesteps)
        ]
        if invalid_steps:
            raise ValueError(
                f"mtd_profile_steps outside [1, {num_sampling_timesteps}]: {invalid_steps}"
            )
        os.makedirs(qnn.mtd_profile_dir, exist_ok=True)

    if qnn.timestep_wise_quant:
        progress_start = opt.quant_progress_start
        progress_end = opt.quant_progress_end
        if progress_start is None or progress_end is None:
            raise ValueError(
                "--timestep_wise_quant requires --quant_progress_start and "
                "--quant_progress_end"
            )
        if not (1 <= progress_start <= progress_end <= num_sampling_timesteps):
            raise ValueError(
                f"invalid progress window [{progress_start}, {progress_end}] for "
                f"{num_sampling_timesteps} sampling steps"
            )
        # Sampling progress is 1..N, while the DDIM loop visits N-1..0.
        qnn.quant_start_t = num_sampling_timesteps - progress_start
        qnn.quant_end_t = num_sampling_timesteps - progress_end
        # The loop must begin in FP and enable quantization only on entry.
        qnn.set_quant_state(False, False)
        logger.info(
            "Experiment-B timestep window: progress [%d, %d] -> internal [%d, %d], "
            "weight_quant=%s, act_quant=%s",
            progress_start,
            progress_end,
            qnn.quant_start_t,
            qnn.quant_end_t,
            qnn.use_weight_quant,
            qnn.use_act_quant,
        )

    metadata = {
        "seed": int(cfg.seed),
        "num_sampling_steps": int(num_sampling_timesteps),
        "timestep_wise_quant": bool(qnn.timestep_wise_quant),
        "weight_quant": bool(qnn.use_weight_quant),
        "act_quant": bool(qnn.use_act_quant),
        "progress_start": opt.get("quant_progress_start", None),
        "progress_end": opt.get("quant_progress_end", None),
        "internal_start": getattr(qnn, "quant_start_t", None),
        "internal_end": getattr(qnn, "quant_end_t", None),
        "quant_ckpt": str(opt.quant_ckpt),
        "prompt_start_index": prompt_start_index,
        "prompt_indices": original_prompt_indices,
        "prompt_count": len(prompts),
        "mtd_profile_steps": sorted(qnn.mtd_profile_steps),
    }
    with open(os.path.join(outpath, "experiment_b_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    sample_idx = 0
    save_dir = opt.save_dir
    os.makedirs(save_dir, exist_ok=True)
    if PRECOMPUTE_TEXT_EMBEDS is not None:
        model_args['precompute_text_embeds'] = torch.load(cfg.precompute_text_embeds)
    print(cfg.batch_size)
    replay_cursor = 0
    for i in range(0, len(prompts), cfg.batch_size):
        qnn.quant_window_trace = []
        batch_prompts = prompts[i : i + cfg.batch_size]
        batch_original_indices = original_prompt_indices[i : i + cfg.batch_size]
        if requested_prompt_indices and replay_original_prompt_rng:
            original_index = int(batch_original_indices[0])
            skipped_prompts = original_index - replay_cursor
            if skipped_prompts < 0:
                raise RuntimeError(
                    f"RNG replay cursor moved backwards: cursor={replay_cursor}, "
                    f"prompt_index={original_index}"
                )
            for _ in range(skipped_prompts):
                torch.randn(
                    1,
                    vae.out_channels,
                    *latent_size,
                    device=device,
                )
                for _ in range(num_sampling_timesteps):
                    torch.randn(
                        2,
                        vae.out_channels,
                        *latent_size,
                        device=device,
                    )
            init_noise = None
            replay_cursor = original_index + 1
        elif requested_prompt_indices:
            per_prompt_noise = []
            for original_index in batch_original_indices:
                noise_generator = torch.Generator(device=device)
                noise_generator.manual_seed(int(cfg.seed) + int(original_index))
                per_prompt_noise.append(torch.randn(
                    1,
                    vae.out_channels,
                    *latent_size,
                    device=device,
                    generator=noise_generator,
                ))
            init_noise = torch.cat(per_prompt_noise, dim=0)
        else:
            noise_generator = torch.Generator(device=device)
            noise_generator.manual_seed(int(cfg.seed) + i)
            init_noise = torch.randn(
                len(batch_prompts),
                vae.out_channels,
                *latent_size,
                device=device,
                generator=noise_generator,
            )
        qnn.mtd_profile_prompt_names = list(batch_prompts)
        qnn.mtd_profile_prompt_indices = list(batch_original_indices)
        if opt.save_init_noise:
            noise_dir = os.path.join(outpath, "init_noise")
            os.makedirs(noise_dir, exist_ok=True)
            torch.save(
                init_noise.detach().float().cpu(),
                os.path.join(noise_dir, f"init_noise_{i:04d}.pt"),
            )
        if PRECOMPUTE_TEXT_EMBEDS is not None:  # also feed in the idxs for saved text_embeds
            model_args['batch_ids'] = torch.tensor(batch_original_indices)
        samples = scheduler.sample(
            qnn,
            text_encoder,
            sampler_type=cfg.sampler,
            z_size=(vae.out_channels, *latent_size),
            prompts=batch_prompts,
            device=device,
            additional_args=model_args,
            init_noise=init_noise,
        )
        if opt.save_final_latent:
            latent_dir = os.path.join(outpath, "final_latents")
            os.makedirs(latent_dir, exist_ok=True)
            for latent_idx, latent in enumerate(samples):
                torch.save(
                    latent.detach().float().cpu(),
                    os.path.join(latent_dir, f"final_latent_{sample_idx + latent_idx:04d}.pt"),
                )
        if opt.save_quant_trace:
            trace_path = os.path.join(outpath, f"quant_trace_batch_{i:04d}.json")
            with open(trace_path, "w") as f:
                json.dump(qnn.quant_window_trace, f, indent=2)
        samples = vae.decode(samples.to(dtype))

        for idx, sample in enumerate(samples):
            print(f"Prompt: {batch_prompts[idx]}")
            if cfg.get("prompt_as_path", False):
                save_path = os.path.join(save_dir, batch_prompts[idx])
            else:
                save_path = os.path.join(save_dir, f"sample_{sample_idx}")
            save_sample(sample, fps=cfg.fps, save_path=save_path)
            sample_idx += 1


if __name__ == "__main__":
    main()
