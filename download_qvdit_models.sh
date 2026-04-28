#!/usr/bin/env bash

set -euo pipefail

DEST_ROOT="/Users/zhouzhongtian/Downloads"
VAE_DEST="${DEST_ROOT}/vae_ckpt"
T5_DEST="${DEST_ROOT}/t5-v1_1-xxl"

echo "Destination root: ${DEST_ROOT}"
mkdir -p "${VAE_DEST}" "${T5_DEST}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3 first."
  exit 1
fi

python3 -m pip install -U "huggingface_hub>=0.25,<1.0"

python3 <<'PY'
from huggingface_hub import snapshot_download

downloads = [
    {
        "repo_id": "stabilityai/sd-vae-ft-ema",
        "local_dir": "/Users/zhouzhongtian/Downloads/vae_ckpt",
        "allow_patterns": [
            "config.json",
            "diffusion_pytorch_model.safetensors",
        ],
    },
    {
        "repo_id": "DeepFloyd/t5-v1_1-xxl",
        "local_dir": "/Users/zhouzhongtian/Downloads/t5-v1_1-xxl",
        "allow_patterns": [
            "config.json",
            "special_tokens_map.json",
            "spiece.model",
            "tokenizer_config.json",
            "pytorch_model.bin.index.json",
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
        ],
    },
]

for item in downloads:
    print(f"\nDownloading {item['repo_id']} -> {item['local_dir']}")
    snapshot_download(
        repo_id=item["repo_id"],
        local_dir=item["local_dir"],
        local_dir_use_symlinks=False,
        allow_patterns=item["allow_patterns"],
    )

print("\nAll downloads completed.")
PY

echo
echo "Downloaded directories:"
echo "  ${VAE_DEST}"
echo "  ${T5_DEST}"
