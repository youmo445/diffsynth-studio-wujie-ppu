#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

WUJIE_REPO = "/data/zsq/diffsynth-studio-wujie-ppu"
if WUJIE_REPO not in sys.path:
    sys.path.insert(0, WUJIE_REPO)

from diffsynth.trainers.utils_robotwin_codex import (
    ROBOTWIN_CAMERA_NAMES,
    ROBOTWIN_DEFAULT_DATA_DIR,
    ROBOTWIN_GRIPPER_BIAS,
    ROBOTWIN_GRIPPER_BIAS_AXIS,
    ROBOTWIN_ORIGINAL_HZ,
    ROBOTWIN_TARGET_HZ,
    build_robotwin_multiview_condition_videos,
    build_robotwin_multiview_rgb,
    episode_path,
    get_downsample_stride,
    get_episode_length,
    make_frame_indices,
    save_video_uint8,
)


def draw_label(frame, label):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 280, 30], fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255))
    return np.array(img)


def build_combined_4col_frames(rgb_frames, action_frames, ray_o_frames, ray_d_frames, frame_indices):
    frames = []
    for i in range(len(rgb_frames)):
        parts = [
            draw_label(rgb_frames[i], f"rgb frame={int(frame_indices[i])}"),
            draw_label(action_frames[i], f"action_map frame={int(frame_indices[i])}"),
            draw_label(ray_o_frames[i], f"ray_o frame={int(frame_indices[i])}"),
            draw_label(ray_d_frames[i], f"ray_d frame={int(frame_indices[i])}"),
        ]
        frames.append(np.concatenate(parts, axis=1))
    return frames


def make_contact_sheet(rgb_frames, action_frames, ray_o_frames, ray_d_frames, out_path, frame_indices):
    rows = []
    for i in [0, len(rgb_frames) // 2, len(rgb_frames) - 1]:
        row = build_combined_4col_frames(
            [rgb_frames[i]],
            [action_frames[i]],
            [ray_o_frames[i]],
            [ray_d_frames[i]],
            [frame_indices[i]],
        )[0]
        rows.append(row)
    sheet = np.concatenate(rows, axis=0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=ROBOTWIN_DEFAULT_DATA_DIR)
    parser.add_argument("--episode", type=str, default="episode0")
    parser.add_argument("--output_dir", type=str, default="/data/zsq/diffsynth-studio-wujie-ppu/outputs/robotwin_visual_debug_codex")
    parser.add_argument("--camera_names", nargs="+", default=ROBOTWIN_CAMERA_NAMES)
    parser.add_argument("--view_height", type=int, default=320)
    parser.add_argument("--view_width", type=int, default=512)
    parser.add_argument("--original_hz", type=int, default=ROBOTWIN_ORIGINAL_HZ)
    parser.add_argument("--target_hz", type=int, default=ROBOTWIN_TARGET_HZ)
    parser.add_argument("--stride", type=int, default=0, help="Optional override. 0 means original_hz // target_hz.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=40)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--traj_radius", type=int, default=22)
    parser.add_argument("--traj_min_radius", type=int, default=12)
    parser.add_argument("--traj_max_radius", type=int, default=40)
    parser.add_argument("--traj_radius_mode", type=str, default="perspective", choices=["constant", "perspective", "depth_norm"])
    parser.add_argument("--gripper_bias", type=float, default=ROBOTWIN_GRIPPER_BIAS)
    parser.add_argument("--gripper_bias_axis", type=str, default=ROBOTWIN_GRIPPER_BIAS_AXIS)
    parser.add_argument("--no_gripper_bias", dest="apply_gripper_bias", action="store_false")
    parser.add_argument("--no_wrist_center_control", dest="wrist_center_control", action="store_false")
    parser.add_argument("--wrist_center_radius", type=int, default=34)
    parser.add_argument("--wrist_center_axis_len", type=int, default=70)
    parser.add_argument("--wrist_other_pixels_per_meter", type=float, default=220.0)
    parser.add_argument("--wrist_other_projection_scale", type=float, default=0.45)
    parser.add_argument("--wrist_other_max_offset_ratio", type=float, default=0.38)
    parser.set_defaults(apply_gripper_bias=True, wrist_center_control=True)
    return parser.parse_args()


def main():
    args = parse_args()
    ep_path = episode_path(args.data_dir, args.episode)
    out_dir = Path(args.output_dir) / ep_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(ep_path, "r") as f:
        length = get_episode_length(f)
        stride = args.stride if args.stride > 0 else get_downsample_stride(args.original_hz, args.target_hz)
        frame_indices = make_frame_indices(length, stride=stride, max_frames=args.max_frames, start=args.start)
        if len(frame_indices) == 0:
            raise RuntimeError("No frames selected.")
        rgb_video = build_robotwin_multiview_rgb(f, frame_indices, args.camera_names, args.view_height, args.view_width)
        action_video, ray_o_video, ray_d_video, abs_actions = build_robotwin_multiview_condition_videos(
            f,
            frame_indices,
            camera_names=args.camera_names,
            view_height=args.view_height,
            view_width=args.view_width,
            traj_radius=args.traj_radius,
            traj_radius_mode=args.traj_radius_mode,
            traj_min_radius=args.traj_min_radius,
            traj_max_radius=args.traj_max_radius,
            apply_gripper_bias=args.apply_gripper_bias,
            gripper_bias=args.gripper_bias,
            gripper_bias_axis=args.gripper_bias_axis,
            wrist_center_control=args.wrist_center_control,
            wrist_center_radius=args.wrist_center_radius,
            wrist_center_axis_len=args.wrist_center_axis_len,
            wrist_other_pixels_per_meter=args.wrist_other_pixels_per_meter,
            wrist_other_projection_scale=args.wrist_other_projection_scale,
            wrist_other_max_offset_ratio=args.wrist_other_max_offset_ratio,
        )

    rgb_path = out_dir / "rgb_video_codex.mp4"
    action_path = out_dir / "action_map_codex.mp4"
    ray_o_path = out_dir / "ray_o_image_codex.mp4"
    ray_d_path = out_dir / "ray_d_image_codex.mp4"
    combined_path = out_dir / "combined_4col_video_codex.mp4"
    sheet_path = out_dir / "preview_sheet_codex.png"
    combined_video = build_combined_4col_frames(rgb_video, action_video, ray_o_video, ray_d_video, frame_indices)
    save_video_uint8(combined_video, combined_path, fps=args.fps)
    save_video_uint8(rgb_video, rgb_path, fps=args.fps)
    save_video_uint8(action_video, action_path, fps=args.fps)
    save_video_uint8(ray_o_video, ray_o_path, fps=args.fps)
    save_video_uint8(ray_d_video, ray_d_path, fps=args.fps)
    make_contact_sheet(rgb_video, action_video, ray_o_video, ray_d_video, sheet_path, frame_indices)
    for name, frames in [
        ("rgb_frame0_codex.png", rgb_video),
        ("action_map_frame0_codex.png", action_video),
        ("ray_o_frame0_codex.png", ray_o_video),
        ("ray_d_frame0_codex.png", ray_d_video),
    ]:
        Image.fromarray(frames[0]).save(out_dir / name)

    meta = {
        "episode_path": str(ep_path),
        "frame_indices": frame_indices.tolist(),
        "camera_names": args.camera_names,
        "original_hz": args.original_hz,
        "target_hz": args.target_hz,
        "downsample_stride": int(stride),
        "view_height": args.view_height,
        "view_width": args.view_width,
        "stacked_height": args.view_height * len(args.camera_names),
        "abs_action_shape": list(abs_actions.shape),
        "abs_action_format": "left xyz quat_xyzw gripper01, right xyz quat_xyzw gripper01",
        "endpose_source_format": "RoboTwin HDF5 stores xyz + quat_wxyz; utils converts to quat_xyzw.",
        "apply_gripper_bias": bool(args.apply_gripper_bias),
        "gripper_bias": float(args.gripper_bias),
        "gripper_bias_axis": args.gripper_bias_axis,
        "wrist_center_control": bool(args.wrist_center_control),
        "wrist_center_radius": int(args.wrist_center_radius),
        "wrist_center_axis_len": int(args.wrist_center_axis_len),
        "wrist_other_pixels_per_meter": float(args.wrist_other_pixels_per_meter),
        "wrist_other_projection_scale": float(args.wrist_other_projection_scale),
        "wrist_other_max_offset_ratio": float(args.wrist_other_max_offset_ratio),
        "outputs": {
            "rgb_video": str(rgb_path),
            "action_map": str(action_path),
            "ray_o_image": str(ray_o_path),
            "ray_d_image": str(ray_d_path),
            "combined_4col_video": str(combined_path),
            "preview_sheet": str(sheet_path),
        },
    }
    meta_path = out_dir / "robotwin_visual_debug_metadata_codex.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
