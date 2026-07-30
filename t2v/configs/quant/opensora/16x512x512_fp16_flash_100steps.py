num_frames = 16
fps = 24 // 3
image_size = (512, 512)

model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    enable_flashattn=True,
    enable_layernorm_kernel=False,
    from_pretrained="/home/zhouchongtian/quantization/Q-VDiT/logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth",
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="/home/zhouchongtian/quantization/models/stabilityai/sd-vae-ft-ema",
    micro_batch_size=128,
)
text_encoder = dict(
    type="t5",
    from_pretrained="/home/zhouchongtian/quantization/models/DeepFloyd",
    local_cache=True,
    save_pretrained="/home/zhouchongtian/quantization/models/DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
)
scheduler = dict(type="iddpm", num_sampling_steps=100, cfg_scale=4.0)
dtype = "fp16"
batch_size = 1
seed = 42
prompt_path = "./t2v/assets/texts/t2v_samples_10.txt"
