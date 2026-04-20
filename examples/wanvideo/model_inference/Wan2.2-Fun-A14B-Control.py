# export PYTHONPATH=/mnt/workspace/zsq/DiffSynth-Studio:$PYTHONPATH

import torch
from diffsynth import save_video,VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from PIL import Image
from modelscope import dataset_snapshot_download

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    model_configs=[
        ModelConfig(path="/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.2-Fun-A14B-Control_high_niose_full/epoch-61.safetensors"),
        ModelConfig(path="/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.2-Fun-A14B-Control_low_niose_full/epoch-61.safetensors"),
        ModelConfig(path="/mnt/workspace/zsq/Wan_model/Wan2.2-Fun-A14B-Control/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="/mnt/workspace/zsq/Wan_model/Wan2.2-Fun-A14B-Control/Wan2.1_VAE.pth"),
    ],
)

# pipe.enable_vram_management()

# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=["data/examples/wan/control_video.mp4", "data/examples/wan/reference_image_girl.png"]
# )

# Control video
H = 320
W = 512
control_video = VideoData("/mnt/workspace/zsq/DiffSynth-Studio/examples/wanvideo/model_inference/data/examples/agi/action_map.mp4", height=H, width=W)
reference_image = Image.open("/mnt/workspace/zsq/DiffSynth-Studio/examples/wanvideo/model_inference/data/examples/agi/first_frame.png").resize((H, W))
video = pipe(
    prompt="机械臂的两个臂执行动作",
    # negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    control_video=control_video, reference_image=reference_image,
    height=H, width=W, num_frames=9,
    num_inference_steps = 50, cfg_scale = 1, 
    seed=1, tiled=True
)
save_video(video, "video.mp4", fps=15, quality=5)