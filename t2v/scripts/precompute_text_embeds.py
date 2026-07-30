import argparse
import os

import torch
from mmengine.config import Config

from opensora.models.text_encoder.t5 import T5Embedder


def load_prompts(prompt_path):
    with open(prompt_path, "r") as file:
        return [line.strip() for line in file if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="model config file path")
    parser.add_argument(
        "--ckpt_path", required=True, help="path to the split STDiT checkpoint"
    )
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--t5_dtype",
        default="fp32",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help="transformers device_map; use 'none' to place the model on --device",
    )
    return parser.parse_args()


def parse_dtype(name):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def load_null_embedding(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    else:
        state_dict = checkpoint
    suffix = "y_embedder.y_embedding"
    key = suffix if suffix in state_dict else next(
        (name for name in state_dict if name.endswith(suffix)), None
    )
    if key is None:
        raise KeyError(f"Cannot find {suffix} in {checkpoint_path}")
    return state_dict[key].float()


def main():
    args = parse_args()
    config = Config.fromfile(args.config)
    text_config = config.text_encoder
    t5_dtype = parse_dtype(args.t5_dtype)
    device_map = None if args.device_map.lower() == "none" else args.device_map
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": t5_dtype,
    }
    if device_map is None:
        # Passing t5_model_kwargs disables T5Embedder's own default device map.
        # Explicitly place both the shared embedding and encoder on the chosen
        # device, otherwise the model remains on CPU while token IDs go to CUDA.
        model_kwargs["device_map"] = {
            "shared": args.device,
            "encoder": args.device,
        }
    else:
        model_kwargs["device_map"] = device_map

    save_pretrained = text_config.get("save_pretrained", None)
    if text_config.get("local_cache", False) and save_pretrained is not None:
        model_name = os.path.basename(save_pretrained.rstrip("/"))
        cache_dir = os.path.dirname(save_pretrained.rstrip("/"))
    else:
        model_name = text_config.from_pretrained
        cache_dir = text_config.from_pretrained

    encoder = T5Embedder(
        device=args.device,
        dir_or_name=model_name,
        local_cache=text_config.get("local_cache", False),
        cache_dir=cache_dir,
        save_pretrained=save_pretrained,
        model_max_length=text_config.model_max_length,
        torch_dtype=t5_dtype,
        t5_model_kwargs=model_kwargs,
    )
    # For an automatic/sharded map, token IDs must enter on the same device as
    # the shared token embedding. Accelerate hooks handle subsequent transfers.
    encoder.device = encoder.model.shared.weight.device
    print(f"T5 input embedding device: {encoder.device}")
    null_embedding = load_null_embedding(args.ckpt_path)
    prompts = load_prompts(args.prompt_path)
    embeddings = []
    masks = []
    with torch.no_grad():
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start:start + args.batch_size]
            conditional, mask = encoder.get_text_embeddings(batch)
            conditional = conditional[:, None]
            unconditional = null_embedding[None].repeat(
                len(batch), 1, 1
            )[:, None].to(conditional.dtype)
            embeddings.append(
                torch.stack(
                    [conditional, unconditional.to(conditional.device)], dim=1
                ).cpu()
            )
            masks.append(mask.cpu())

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    payload = {
        "y": torch.cat(embeddings, dim=0),
        "mask": torch.cat(masks, dim=0),
    }
    torch.save(payload, args.save_path)
    print(
        f"Saved {len(prompts)} prompt embeddings to {args.save_path}: "
        f"y={tuple(payload['y'].shape)}, mask={tuple(payload['mask'].shape)}"
    )


if __name__ == "__main__":
    main()
