#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from arxx5_dataset5_wancontrolmultiview import Arxx5Dataset5Wancontrolmultiview, write_video_rgb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate Arxx5Dataset5Wancontrolmultiview by saving RGB/control/ray_o/ray_d into one video."
    )
    p.add_argument("--dataset-root", type=str, default="/mnt/workspace/zsq/agx")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--episode-limit", type=int, default=2)
    p.add_argument("--num-frames", type=int, default=9)
    p.add_argument("--target-hz", type=float, default=5)
    p.add_argument("--resize-h", type=int, default=320)
    p.add_argument("--resize-w", type=int, default=512)
    p.add_argument("--output-video", type=Path, default=Path("/mnt/workspace/zsq/agx_debug_sample.mp4"))
    p.add_argument("--fps", type=float, default=5)
    p.add_argument("--camera-axis-mode", type=str, default="fixed", choices=["fixed", "identity", "frames3d"])
    p.add_argument("--traj-radius-mode", type=str, default="perspective", choices=["constant", "perspective", "depth_norm"])
    p.add_argument("--traj-radius", type=int, default=40)
    p.add_argument("--head-left-calib", type=str, default="/mnt/workspace/zsq/outputs/calib_eye_to_hand_head_left/result_eye_to_hand.json")
    p.add_argument("--head-right-calib", type=str, default="/mnt/workspace/zsq/outputs/calib_eye_to_hand_head_right/result_eye_to_hand.json")
    p.add_argument("--left-calib", type=str, default="/mnt/workspace/zsq/outputs/calib_eye_in_hand_left/result_eye_in_hand.json")
    p.add_argument("--right-calib", type=str, default="/mnt/workspace/zsq/outputs/calib_eye_in_hand_right/result_eye_in_hand.json")
    return p.parse_args()


def add_title(img: Image.Image, title: str, title_h: int = 32) -> Image.Image:
    arr = np.asarray(img, dtype=np.uint8)
    canvas = Image.new("RGB", (arr.shape[1], arr.shape[0] + title_h), (20, 20, 20))
    canvas.paste(Image.fromarray(arr), (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(255, 255, 255))
    return canvas


def main() -> None:
    args = parse_args()
    dataset = Arxx5Dataset5Wancontrolmultiview(
        base_path=args.dataset_root,
        num_frames=args.num_frames,
        target_hz=args.target_hz,
        episode_limit=args.episode_limit,
        resize_to=(args.resize_h, args.resize_w),
        dataset_type="vace",
        output_raymap=True,
        raymap_mode="image",
        camera_axis_mode=args.camera_axis_mode,
        traj_radius_mode=args.traj_radius_mode,
        traj_radius=args.traj_radius,
        head_left_calib=args.head_left_calib,
        head_right_calib=args.head_right_calib,
        left_calib=args.left_calib,
        right_calib=args.right_calib,
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset returned zero samples. Check paths, calibration files, videos, and parquet files.")

    sample = dataset[args.sample_index]
    video = sample["video"]
    control_video = sample["vace_video"]
    ray_o = sample["ray_map_o"]
    ray_d = sample["ray_map_d"]

    print("[INFO] sample meta:", sample.get("meta"))
    print("[INFO] keys:", sorted(sample.keys()))
    print("[INFO] video frames:", len(video), "frame size:", video[0].size)

    frames = []
    for i in range(len(video)):
        parts = [
            np.asarray(add_title(video[i], "video"), dtype=np.uint8),
            np.asarray(add_title(control_video[i], "control_video / traj_map"), dtype=np.uint8),
            np.asarray(add_title(ray_o[i], "ray_map_o"), dtype=np.uint8),
            np.asarray(add_title(ray_d[i], "ray_map_d"), dtype=np.uint8),
        ]
        frames.append(np.concatenate(parts, axis=1))

    write_video_rgb(frames, args.output_video, fps=args.fps)
    print(f"[OK] saved visualization: {args.output_video}")


if __name__ == "__main__":
    main()
