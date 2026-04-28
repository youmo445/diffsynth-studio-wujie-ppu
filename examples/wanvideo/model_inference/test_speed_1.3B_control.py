"""
Benchmark inference speed (FPS) on ONE AgiBot validation episode only.
Measures generated-frame throughput for Wan2.2-A14B-Control.

Example:
python benchmark_one_episode_fps.py \
    --val_path /mnt/workspace/zsq/Agibotsubset/val \
    --episode_idx 0
"""

import os
import time
import math
import json
import h5py
import argparse
import numpy as np
import torch
from PIL import Image
from collections import defaultdict
from decord import VideoReader, cpu


# =====================================================
# Find one episode
# =====================================================

def discover_episodes(val_path):
    groups = defaultdict(list)

    for name in sorted(os.listdir(val_path)):
        p = os.path.join(val_path, name)
        if os.path.isdir(p):
            key = "-".join(name.split("-")[:2])
            groups[key].append(p)

    episodes = []
    for k in sorted(groups.keys()):
        episodes.append((k, sorted(groups[k])))

    return episodes


# =====================================================
# Minimal data loader
# =====================================================

def load_episode(ep_paths, step, H, W):
    pos_all = []

    for sub in ep_paths:
        with h5py.File(os.path.join(sub, "proprio_stats.h5"), "r") as f:
            pos = f["state/end/position"][:]
        pos_all.append(pos)

    pos_all = np.concatenate(pos_all, axis=0)
    total_frames = len(np.arange(0, len(pos_all), step))

    vr = VideoReader(os.path.join(ep_paths[0], "head_color.mp4"), ctx=cpu(0))
    first = vr[0].asnumpy()
    first = Image.fromarray(first).resize((W, H))

    # fake control video (测速不关心内容)
    dummy = Image.new("RGB", (W, H), (128, 128, 128))

    return first, dummy, total_frames


# =====================================================
# Load model
# =====================================================

def load_pipe(paths):
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

    cfg = [ModelConfig(path=p) for p in paths]

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=cfg,
    )
    return pipe


# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--val_path", type=str,
        default="/mnt/workspace/zsq/Agibotsubset/val")

    parser.add_argument("--episode_idx", type=int, default=0)

    parser.add_argument("--predict_frames", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=5)

    parser.add_argument("--original_hz", type=int, default=30)
    parser.add_argument("--target_hz", type=int, default=5)

    parser.add_argument("--H", type=int, default=320)
    parser.add_argument("--W", type=int, default=512)

    parser.add_argument("--model_paths", nargs=4, default=[
        "/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.1-Fun-1.3B-Control/epoch-49.safetensors",
        "/mnt/workspace/zsq/Wan_model/Wan2.1-Fun-1.3B-Control-deprated/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "/mnt/workspace/zsq/Wan_model/Wan2.1-Fun-1.3B-Control-deprated/models_t5_umt5-xxl-enc-bf16.pth",
        "/mnt/workspace/zsq/Wan_model/Wan2.1-Fun-1.3B-Control-deprated/Wan2.1_VAE.pth"
    ])

    args = parser.parse_args()

    step = args.original_hz // args.target_hz

    # episode
    episodes = discover_episodes(args.val_path)
    ep_key, ep_paths = episodes[args.episode_idx]

    print("Benchmark episode:", ep_key)

    # load data
    first_frame, dummy_map, total_frames = load_episode(
        ep_paths, step, args.H, args.W
    )

    # load model
    pipe = load_pipe(args.model_paths)

    predict_total = total_frames - 1
    num_chunks = math.ceil(predict_total / args.predict_frames)

    current = first_frame
    generated_frames = 0

    torch.cuda.synchronize()
    t0 = time.time()

    for i in range(num_chunks):

        valid = min(
            args.predict_frames,
            predict_total - i * args.predict_frames
        )

        control_video = [dummy_map] * (1 + args.predict_frames)

        out = pipe(
            prompt="机械臂的两个臂执行动作",
            control_video=control_video,
            reference_image=current,
            height=args.H,
            width=args.W,
            num_frames=1 + args.predict_frames,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=1.0,
            tiled=True,
            seed=42,
        )

        current = out[valid]
        generated_frames += valid

    torch.cuda.synchronize()
    t1 = time.time()

    sec = t1 - t0
    fps = generated_frames / sec

    print("=" * 50)
    print("Generated frames :", generated_frames)
    print("Total time (s)   :", round(sec, 3))
    print("Inference FPS    :", round(fps, 3))
    print("=" * 50)


if __name__ == "__main__":
    main()