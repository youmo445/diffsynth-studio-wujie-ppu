#!/usr/bin/env python3
import argparse
import json
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
    build_robotwin_abs_actions,
    build_robotwin_multiview_condition_videos,
    build_robotwin_multiview_rgb,
    episode_path,
    extrinsic_cv_to_w2c,
    get_downsample_stride,
    get_episode_length,
    make_frame_indices,
    pose7_to_matrix,
    read_robotwin_rgb,
    resize_intrinsic,
    save_video_uint8,
)


def project_action_bases(abs_actions, w2c, intrinsic):
    w2c = np.asarray(w2c, dtype=np.float32)
    if w2c.ndim == 2:
        w2c = np.repeat(w2c[None], abs_actions.shape[0], axis=0)
    pose_l = pose7_to_matrix(abs_actions[:, 0:7])
    pose_r = pose7_to_matrix(abs_actions[:, 8:15])
    pts_l = np.matmul(w2c, pose_l)[:, :3, 3]
    pts_r = np.matmul(w2c, pose_r)[:, :3, 3]
    intr = intrinsic.reshape(1, 3, 3)
    uv_l = np.matmul(intr, pts_l[:, :, None])
    uv_l = (uv_l[:, :2, 0] / (pts_l[:, 2:3] + 1e-8))
    uv_r = np.matmul(intr, pts_r[:, :, None])
    uv_r = (uv_r[:, :2, 0] / (pts_r[:, 2:3] + 1e-8))
    return uv_l, pts_l[:, 2], uv_r, pts_r[:, 2]


def visibility_stats(h5_file, frame_indices, abs_actions, camera_names, view_height, view_width):
    out = {}
    for cam in camera_names:
        sample = read_robotwin_rgb(h5_file, cam, int(frame_indices[0]))
        src_h, src_w = sample.shape[:2]
        intrinsic = resize_intrinsic(
            h5_file[f"observation/{cam}/intrinsic_cv"][int(frame_indices[0])],
            src_h,
            src_w,
            view_height,
            view_width,
        )
        w2c = extrinsic_cv_to_w2c(h5_file[f"observation/{cam}/extrinsic_cv"][frame_indices])
        uv_l, z_l, uv_r, z_r = project_action_bases(abs_actions, w2c, intrinsic)
        vis_l = (z_l > 1e-4) & (uv_l[:, 0] >= 0) & (uv_l[:, 0] < view_width) & (uv_l[:, 1] >= 0) & (uv_l[:, 1] < view_height)
        vis_r = (z_r > 1e-4) & (uv_r[:, 0] >= 0) & (uv_r[:, 0] < view_width) & (uv_r[:, 1] >= 0) & (uv_r[:, 1] < view_height)
        out[cam] = {
            "left_visible": int(vis_l.sum()),
            "right_visible": int(vis_r.sum()),
            "left_visible_ratio": float(vis_l.mean()),
            "right_visible_ratio": float(vis_r.mean()),
            "both_visible": int((vis_l & vis_r).sum()),
            "both_visible_ratio": float((vis_l & vis_r).mean()),
            "left_uv_mean_visible": uv_l[vis_l].mean(axis=0).tolist() if vis_l.any() else None,
            "right_uv_mean_visible": uv_r[vis_r].mean(axis=0).tolist() if vis_r.any() else None,
        }
    return out


def draw_variant_label(frame, label):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 520, 34], fill=(0, 0, 0))
    draw.text((8, 9), label, fill=(255, 255, 255))
    return np.array(img)


def make_variant_sheet(rgb_frames, variant_to_action, frame_indices, out_path):
    sample_ids = [0, len(rgb_frames) // 2, len(rgb_frames) - 1]
    rows = []
    for i in sample_ids:
        parts = [draw_variant_label(rgb_frames[i], f"rgb frame={int(frame_indices[i])}")]
        for key, frames in variant_to_action.items():
            parts.append(draw_variant_label(frames[i], key))
        rows.append(np.concatenate(parts, axis=1))
    sheet = np.concatenate(rows, axis=0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=ROBOTWIN_DEFAULT_DATA_DIR)
    parser.add_argument("--episode", type=str, default="episode0")
    parser.add_argument("--output_dir", type=str, default="/data/zsq/diffsynth-studio-wujie-ppu/outputs/robotwin_gripper_bias_sweep_codex")
    parser.add_argument("--camera_names", nargs="+", default=ROBOTWIN_CAMERA_NAMES)
    parser.add_argument("--view_height", type=int, default=320)
    parser.add_argument("--view_width", type=int, default=512)
    parser.add_argument("--original_hz", type=int, default=30)
    parser.add_argument("--target_hz", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=40)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--bias", type=float, default=ROBOTWIN_GRIPPER_BIAS)
    parser.add_argument("--axes", nargs="+", default=["x", "-x", "y", "-y", "z", "-z"])
    return parser.parse_args()


def main():
    args = parse_args()
    ep = episode_path(args.data_dir, args.episode)
    out_dir = Path(args.output_dir) / ep.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    stride = get_downsample_stride(args.original_hz, args.target_hz)
    with h5py.File(ep, "r") as f:
        length = get_episode_length(f)
        frame_indices = make_frame_indices(length, stride=stride, max_frames=args.max_frames, start=args.start)
        rgb_video = build_robotwin_multiview_rgb(f, frame_indices, args.camera_names, args.view_height, args.view_width)
        variant_to_action = {}
        all_stats = {}
        for axis in args.axes:
            action_video, _, _, abs_actions = build_robotwin_multiview_condition_videos(
                f,
                frame_indices,
                camera_names=args.camera_names,
                view_height=args.view_height,
                view_width=args.view_width,
                apply_gripper_bias=True,
                gripper_bias=args.bias,
                gripper_bias_axis=axis,
            )
            key = f"bias_{args.bias:g}_{axis}"
            variant_to_action[key] = action_video
            save_video_uint8(action_video, out_dir / f"action_map_{key}_codex.mp4", fps=args.fps)
            all_stats[key] = visibility_stats(f, frame_indices, abs_actions, args.camera_names, args.view_height, args.view_width)

    make_variant_sheet(rgb_video, variant_to_action, frame_indices, out_dir / "bias_axis_sweep_sheet_codex.png")
    meta = {
        "episode_path": str(ep),
        "frame_indices": frame_indices.tolist(),
        "bias": args.bias,
        "axes": args.axes,
        "stats": all_stats,
        "sheet": str(out_dir / "bias_axis_sweep_sheet_codex.png"),
    }
    (out_dir / "bias_axis_sweep_metadata_codex.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
