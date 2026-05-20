#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from arxx5_dataset4_wancontrolmultiview import Arxx5Dataset4Wancontrolmultiview, write_video_rgb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, default="/mnt/workspace/zsq/agx")
    p.add_argument("--sample-index", type=int, default=100)
    p.add_argument("--output-video", type=Path, default="/mnt/workspace/zsq/agx_debug_sample.mp4")
    p.add_argument("--num-frames", type=int, default=281)
    p.add_argument("--target-hz", type=float, default=30)
    p.add_argument("--episode-limit", type=int, default=None)
    p.add_argument("--resize-h", type=int, default=168)
    p.add_argument("--resize-w", type=int, default=224)
    p.add_argument("--camera-axis-mode", type=str, default="identity", choices=["fixed", "identity", "frames3d"])
    p.add_argument("--head-left-calib", type=str, default="/mnt/workspace/zsq/outputs/calib_eye_to_hand_head_left/result_eye_to_hand.json")
    p.add_argument("--head-right-calib", type=str, default="/mnt/workspace/zsq/outputs/calib_eye_to_hand_head_right/result_eye_to_hand.json")
    p.add_argument("--left-calib", type=str, default="/mnt/workspace/zsq/outputs/calib_eye_in_hand_left/result_eye_in_hand.json")
    p.add_argument("--right-calib", type=str, default="/mnt/workspace/zsq/outputs/calib_eye_in_hand_right/result_eye_in_hand.json")
    return p.parse_args()


def add_title(img: Image.Image, title: str, bar_h: int = 28) -> Image.Image:
    img = img.convert("RGB")
    out = Image.new("RGB", (img.width, img.height + bar_h), (20, 20, 20))
    out.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(out)
    draw.text((8, 7), title, fill=(255, 255, 255))
    return out


def main():
    args = parse_args()
    dataset = Arxx5Dataset4Wancontrolmultiview(
        base_path=args.dataset_root,
        num_frames=args.num_frames,
        target_hz=args.target_hz,
        episode_limit=args.episode_limit,
        resize_to=(args.resize_h, args.resize_w),
        output_raymap=True,
        raymap_mode="image",
        dataset_type="vace",
        camera_axis_mode=args.camera_axis_mode,
        head_left_calib=args.head_left_calib,
        head_right_calib=args.head_right_calib,
        left_calib=args.left_calib,
        right_calib=args.right_calib,
    )
    sample = dataset[args.sample_index]
    print("sample meta:", sample["meta"])

    video = sample["video"]
    control = sample["vace_video"]
    ray_o = sample["ray_map_o"]
    ray_d = sample["ray_map_d"]

    frames = []
    for i in range(len(video)):
        cols = [
            add_title(video[i], "video"),
            add_title(control[i], "control_video / traj_map"),
            add_title(ray_o[i], "ray_map_o(center-base)"),
            add_title(ray_d[i], "ray_map_d(center-base)"),
        ]
        h = max(c.height for c in cols)
        padded = []
        for c in cols:
            if c.height < h:
                canvas = Image.new("RGB", (c.width, h), (0, 0, 0))
                canvas.paste(c, (0, 0))
                c = canvas
            padded.append(c)
        frame = Image.new("RGB", (sum(c.width for c in padded), h), (0, 0, 0))
        x = 0
        for c in padded:
            frame.paste(c, (x, 0))
            x += c.width
        frames.append(np.asarray(frame))

    write_video_rgb(frames, args.output_video, fps=args.target_hz)
    print(f"saved: {args.output_video}")


if __name__ == "__main__":
    main()
