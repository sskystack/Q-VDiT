import argparse
import os

import torch
from mmengine.config import Config

from opensora.models.text_encoder.t5 import T5Embedder


def load_prompts(prompt_path):
    with open(prompt_path, "r") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_null_embedding(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    key = "y_embedder.y_embedding"
    if key not in state_dict:
        matches = [name for name in state_dict if name.endswith(key)]
        if not matches:
            raise KeyError(f"Cannot find {key} in {checkpoint_path}")
        key = matches[0]
    return state_dict[key].float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--batch_size", default=2, type=int)
    args = parser.parse_args()

    config = Config.fromfile(args.config)
    text_config = config.text_encoder
    save_pretrained = text_config.get("save_pretrained")
    dir_or_name = os.path.basename(save_pretrained.rstrip("/"))
    cache_dir = os.path.dirname(save_pretrained.rstrip("/"))
    t5 = T5Embedder(
        device="cuda:0",
        dir_or_name=dir_or_name,
        local_cache=True,
        cache_dir=cache_dir,
        save_pretrained=save_pretrained,
        model_max_length=text_config.model_max_length,
        torch_dtype=torch.float32,
        t5_model_kwargs={"low_cpu_mem_usage": True, "torch_dtype": torch.float32, "device_map": "auto"},
    )
    null_embedding = load_null_embedding(args.ckpt_path)
    prompts = load_prompts(args.prompt_path)
    embeddings = []
    masks = []
    with torch.no_grad():
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            conditional, mask = t5.get_text_embeddings(batch)
            conditional = conditional[:, None]
            unconditional = null_embedding[None].repeat(len(batch), 1, 1)[:, None].to(conditional)
            embeddings.append(torch.stack((conditional, unconditional), dim=1).cpu())
            masks.append(mask.cpu())
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save({"y": torch.cat(embeddings), "mask": torch.cat(masks)}, args.save_path)
    print(f"Saved embeddings for {len(prompts)} prompts to {args.save_path}")


if __name__ == "__main__":
    main()
