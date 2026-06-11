import argparse
import os

import torch
from mmengine.config import Config

from opensora.models.text_encoder.t5 import T5Embedder


def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="model config file path")
    parser.add_argument("--ckpt_path", required=True, help="path to split STDiT checkpoint")
    parser.add_argument("--prompt_path", required=True, help="path to prompt txt file")
    parser.add_argument("--save_path", required=True, help="path to save precomputed text embeddings")
    parser.add_argument("--batch_size", default=8, type=int, help="T5 encoding batch size")
    parser.add_argument("--device", default="cuda", help="device for token tensors")
    parser.add_argument("--t5_dtype", default="fp32", choices=["fp16", "bf16", "fp32"], help="T5 dtype")
    parser.add_argument(
        "--device_map",
        default="auto",
        help="transformers device_map for T5; use 'none' to place the model on --device",
    )
    return parser.parse_args()


def parse_dtype(name):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def load_null_embedding(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    else:
        state_dict = ckpt
    key = "y_embedder.y_embedding"
    if key not in state_dict:
        matches = [name for name in state_dict if name.endswith(key)]
        if not matches:
            raise KeyError(f"Cannot find {key} in {ckpt_path}")
        key = matches[0]
    return state_dict[key].float()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    text_cfg = cfg.text_encoder
    t5_dtype = parse_dtype(args.t5_dtype)
    device_map = None if args.device_map.lower() == "none" else args.device_map
    t5_model_kwargs = {"low_cpu_mem_usage": True, "torch_dtype": t5_dtype}
    if device_map is not None:
        t5_model_kwargs["device_map"] = device_map

    save_pretrained = text_cfg.get("save_pretrained", None)
    if text_cfg.get("local_cache", False) and save_pretrained is not None:
        dir_or_name = os.path.basename(save_pretrained.rstrip("/"))
        cache_dir = os.path.dirname(save_pretrained.rstrip("/"))
    else:
        dir_or_name = text_cfg.from_pretrained
        cache_dir = text_cfg.from_pretrained

    t5 = T5Embedder(
        device=args.device,
        dir_or_name=dir_or_name,
        local_cache=text_cfg.get("local_cache", False),
        cache_dir=cache_dir,
        save_pretrained=save_pretrained,
        model_max_length=text_cfg.model_max_length,
        torch_dtype=t5_dtype,
        t5_model_kwargs=t5_model_kwargs,
    )
    null_embedding = load_null_embedding(args.ckpt_path)

    prompts = load_prompts(args.prompt_path)
    ys = []
    masks = []
    with torch.no_grad():
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            caption_embs, emb_masks = t5.get_text_embeddings(batch)
            cond_y = caption_embs[:, None]
            null_y = null_embedding[None].repeat(len(batch), 1, 1)[:, None].to(cond_y.dtype)
            y = torch.stack([cond_y, null_y.to(cond_y.device)], dim=1).cpu()
            ys.append(y)
            masks.append(emb_masks.cpu())

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save({"y": torch.cat(ys, dim=0), "mask": torch.cat(masks, dim=0)}, args.save_path)
    print(f"Saved {len(prompts)} text embeddings to {args.save_path}")


if __name__ == "__main__":
    main()
