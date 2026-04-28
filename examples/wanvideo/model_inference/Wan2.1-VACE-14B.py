import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    model_configs=[
        ModelConfig(path=["/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/diffusion_pytorch_model-00001-of-00007.safetensors",
            "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/diffusion_pytorch_model-00002-of-00007.safetensors",
            "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/diffusion_pytorch_model-00003-of-00007.safetensors",
            "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/diffusion_pytorch_model-00004-of-00007.safetensors",
            "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/diffusion_pytorch_model-00005-of-00007.safetensors",
            "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/diffusion_pytorch_model-00006-of-00007.safetensors",
            "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/diffusion_pytorch_model-00007-of-00007.safetensors",],),
        ModelConfig(path="/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/Wan2.1_VAE.pth"),
    ],
)


pipe.enable_vram_management()
H = 320
W = 512
control_video = VideoData("/mnt/workspace/zsq/DiffSynth-Studio/examples/wanvideo/model_inference/data/examples/agi/action_map_trim.mp4", height=H, width=W)
reference_image = Image.open("/mnt/workspace/zsq/DiffSynth-Studio/examples/wanvideo/model_inference/data/examples/agi/first_frame.png").resize((W, H))
# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=["data/examples/wan/depth_video.mp4", "data/examples/wan/cat_fightning.jpg"]
# )

# Depth video -> Video
# control_video = VideoData("data/examples/wan/depth_video.mp4", height=480, width=832)
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     vace_video=control_video,
#     seed=1, tiled=True
# )
# save_video(video, "video1_14b.mp4", fps=15, quality=5)

# # Reference image -> Video
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     vace_reference_image=Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480)),
#     seed=1, tiled=True
# )
# save_video(video, "video2_14b.mp4", fps=15, quality=5)

# Depth video + Reference image -> Video
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     vace_video=control_video,
#     vace_reference_image=Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480)),
#     seed=1, tiled=True
# )
video = pipe(
    prompt="机械臂的两个臂执行动作",
    # negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=control_video,
    vace_reference_image=reference_image,
    height=H, width=W, num_frames=9,
    num_inference_steps = 50, cfg_scale = 1, 
    seed=1, tiled=True
)
save_video(video, "video3_14b.mp4", fps=15, quality=5)
