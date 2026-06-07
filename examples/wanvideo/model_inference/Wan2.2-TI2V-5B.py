import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    model_configs=[
        ModelConfig(path=["/mnt/workspace/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
            "/mnt/workspace/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
            "/mnt/workspace/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"]),
        ModelConfig(path="/mnt/workspace/zsq/Wan_model/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="/mnt/workspace/zsq/Wan_model/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
    ],
)
pipe.enable_vram_management()

# # Text-to-video
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=0, tiled=True,
#     height=704, width=1248,
#     num_frames=121,
# )
# save_video(video, "video1.mp4", fps=15, quality=5)

# Image-to-video
# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=["data/examples/wan/cat_fightning.jpg"]
# )
input_image = Image.open("/mnt/workspace/zsq/DiffSynth-Studio/examples/wanvideo/model_inference/data/examples/arxx5/1.png")
video = pipe(
    prompt="三视角双臂机械臂桌面操作视频。输入图从上到下分别是头部相机、左腕相机、右腕相机视角。桌面中央有一个绿色盒子，左侧有一个橙色方块，右侧有一个蓝色方块。左臂抓取橙色方块并放入绿色盒子，然后右臂抓取蓝色方块并放入绿色盒子。机械臂、桌面、盒子和方块在整个视频中保持一致，三路视角内容相互对应。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=False,
    height=480, width=224,
    input_image=input_image,
    num_inference_steps=24,
    num_frames=25,
)
save_video(video, "video2.mp4", fps=15, quality=5)
