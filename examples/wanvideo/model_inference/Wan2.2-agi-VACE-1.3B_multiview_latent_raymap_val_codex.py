"""
Multi-view Wan VACE inference aligned with the current multiview training logic.

This script keeps trajectory projection and optional raymap generation consistent
with diffsynth/trainers/utils_codex.py.

Example:
    cd /mnt/workspace/zsq/DiffSynth-Studio
    python examples/wanvideo/model_inference/Wan2.2-agi-VACE-1.3B_multiview_latent_raymap_val_codex.py \
        --val_path /mnt/workspace/zsq/Agi2024subset_split/val \
        --output_dir /mnt/workspace/zsq/Agi2024subset_split/val_gen_multiview
"""

import argparse
import json
import math
import multiprocessing as mp
import os

import h5py
import imageio.v3 as iio
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image

from diffsynth import load_state_dict, save_video
from diffsynth.trainers.utils_codex import (
    apply_eef2cam_visual_rotation,
    generate_traj_map,
)


def discover_episodes(base_path):
    proprio_base = os.path.join(base_path, "proprio_stats")
    if os.path.isdir(proprio_base):
        episodes = []
        for task in sorted(os.listdir(proprio_base)):
            task_path = os.path.join(proprio_base, task)
            if not os.path.isdir(task_path):
                continue
            for ep in sorted(os.listdir(task_path)):
                h5_path = os.path.join(task_path, ep, "proprio_stats.h5")
                if not os.path.exists(h5_path):
                    continue
                episodes.append(
                    {
                        "episode_key": f"{task}-{ep}",
                        "layout": "split",
                        "h5_path": h5_path,
                        "video_dir": os.path.join(base_path, "observations", task, ep, "videos"),
                        "camera_dir": os.path.join(base_path, "parameters", task, ep, "parameters", "camera"),
                    }
                )
        return episodes

    episode_dict = {}
    for name in sorted(os.listdir(base_path)):
        path = os.path.join(base_path, name)
        if os.path.isdir(path):
            key = "-".join(name.split("-")[:2])
            episode_dict.setdefault(key, []).append(path)
    return [
        {
            "episode_key": key,
            "layout": "chunk",
            "chunk_paths": sorted(paths),
        }
        for key, paths in sorted(episode_dict.items())
    ]


def compute_abs_action(pos, quat, grip):
    out = np.zeros((pos.shape[0], 16), dtype=np.float32)
    out[:, 0:3] = pos[:, 0]
    out[:, 3:7] = apply_eef2cam_visual_rotation(quat[:, 0], "left")
    out[:, 7] = grip[:, 0]
    out[:, 8:11] = pos[:, 1]
    out[:, 11:15] = apply_eef2cam_visual_rotation(quat[:, 1], "right")
    out[:, 15] = grip[:, 1]
    return out


def load_camera_params(camera_dir, cam_name, ds_idx):
    intrinsic_path = os.path.join(camera_dir, f"{cam_name}_intrinsic_params.json")
    extrinsic_path = os.path.join(camera_dir, f"{cam_name}_extrinsic_params_aligned.json")

    with open(intrinsic_path, "r", encoding="utf-8") as f:
        info = json.load(f)["intrinsic"]
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = info["fx"]
    intrinsic[1, 1] = info["fy"]
    intrinsic[0, 2] = info["ppx"]
    intrinsic[1, 2] = info["ppy"]

    with open(extrinsic_path, "r", encoding="utf-8") as f:
        extr_list = json.load(f)
    c2w_all = []
    for item in extr_list:
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = np.array(item["extrinsic"]["rotation_matrix"], dtype=np.float32)
        mat[:3, 3] = np.array(item["extrinsic"]["translation_vector"], dtype=np.float32)
        c2w_all.append(mat)
    c2w_all = np.stack(c2w_all, axis=0)
    valid_idx = np.clip(ds_idx, 0, c2w_all.shape[0] - 1)
    c2w_seq = c2w_all[valid_idx]
    w2c_seq = np.linalg.inv(c2w_seq).astype(np.float32)
    return {"intrinsic": intrinsic, "w2c": w2c_seq, "c2w": c2w_seq}


def build_episode_info(ep, downsample_step, camera_names):
    if ep["layout"] == "split":
        with h5py.File(ep["h5_path"], "r") as f:
            pos = f["state/end/position"][:]
            quat = f["state/end/orientation"][:]
            grip = f["state/effector/position"][:]

        total_raw = pos.shape[0]
        grip = grip.reshape(total_raw, 2, -1)[..., 0] if grip.ndim == 3 else grip
        ds_idx = np.arange(0, total_raw, downsample_step)
        abs_actions = compute_abs_action(pos[ds_idx], quat[ds_idx], grip[ds_idx])
        cameras = {cam: load_camera_params(ep["camera_dir"], cam, ds_idx) for cam in camera_names}
        return {
            "episode_key": ep["episode_key"],
            "layout": "split",
            "video_dir": ep["video_dir"],
            "ds_indices": ds_idx,
            "T_ds": len(ds_idx),
            "abs_actions": abs_actions,
            "cameras": cameras,
        }

    chunk_paths = ep["chunk_paths"]
    pos_list, quat_list, grip_list, chunk_frame_counts = [], [], [], []
    for chunk_path in chunk_paths:
        h5_path = os.path.join(chunk_path, "proprio_stats.h5")
        with h5py.File(h5_path, "r") as f:
            pos = f["state/end/position"][:]
            quat = f["state/end/orientation"][:]
            grip = f["state/effector/position"][:]
        chunk_frame_counts.append(pos.shape[0])
        pos_list.append(pos)
        quat_list.append(quat)
        grip_list.append(grip.reshape(pos.shape[0], 2, -1)[..., 0] if grip.ndim == 3 else grip)

    pos_all = np.concatenate(pos_list, axis=0)
    quat_all = np.concatenate(quat_list, axis=0)
    grip_all = np.concatenate(grip_list, axis=0)
    ds_idx = np.arange(0, pos_all.shape[0], downsample_step)
    abs_actions = compute_abs_action(pos_all[ds_idx], quat_all[ds_idx], grip_all[ds_idx])
    cameras = {cam: load_camera_params(chunk_paths[0], cam, ds_idx) for cam in camera_names}
    return {
        "episode_key": ep["episode_key"],
        "layout": "chunk",
        "chunk_paths": chunk_paths,
        "chunk_frame_counts": chunk_frame_counts,
        "ds_indices": ds_idx,
        "T_ds": len(ds_idx),
        "abs_actions": abs_actions,
        "cameras": cameras,
    }


def read_video_frame(video_path, frame_idx):
    try:
        frame = iio.imread(video_path, index=frame_idx)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        return frame
    except Exception:
        vr = VideoReader(video_path, ctx=cpu(0))
        frame = vr[min(frame_idx, len(vr) - 1)].asnumpy()
        del vr
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        return frame


def load_first_multiview_frame(ep_info, camera_names, view_height, view_width):
    frames = []
    if ep_info["layout"] == "split":
        orig_idx = int(ep_info["ds_indices"][0])
        for cam in camera_names:
            video_path = os.path.join(ep_info["video_dir"], f"{cam}_color.mp4")
            frame = read_video_frame(video_path, orig_idx)
            frames.append(np.array(Image.fromarray(frame).resize((view_width, view_height), Image.BILINEAR)))
        return Image.fromarray(np.concatenate(frames, axis=0))

    chunk_cum = np.cumsum([0] + ep_info["chunk_frame_counts"])
    orig_idx = int(ep_info["ds_indices"][0])
    cid = int(np.searchsorted(chunk_cum[1:], orig_idx, side="right"))
    local_idx = orig_idx - int(chunk_cum[cid])
    for cam in camera_names:
        video_path = os.path.join(ep_info["chunk_paths"][cid], f"{cam}_color.mp4")
        frame = read_video_frame(video_path, local_idx)
        frames.append(np.array(Image.fromarray(frame).resize((view_width, view_height), Image.BILINEAR)))
    return Image.fromarray(np.concatenate(frames, axis=0))


def generate_raymap(intrinsic, c2w, height, width, ray_o_vmin, ray_o_vmax, ray_d_vmin, ray_d_vmax):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    xs = np.arange(width, dtype=np.float32) + 0.5
    ys = np.arange(height, dtype=np.float32) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    dirs = np.stack(
        [
            (xx - cx) / fx,
            (yy - cy) / fy,
            np.ones_like(xx),
        ],
        axis=-1,
    )

    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]
    rays_d = dirs @ rotation.T
    rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
    rays_o = np.broadcast_to(translation, rays_d.shape).copy()

    ray_o = np.clip((rays_o - ray_o_vmin) / (ray_o_vmax - ray_o_vmin + 1e-8), 0.0, 1.0)
    ray_d = np.clip((rays_d - ray_d_vmin) / (ray_d_vmax - ray_d_vmin + 1e-8), 0.0, 1.0)
    return (ray_o * 255).astype(np.uint8), (ray_d * 255).astype(np.uint8)


def generate_raw_raymap(intrinsic, c2w, height, width):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    xs = np.arange(width, dtype=np.float32) + 0.5
    ys = np.arange(height, dtype=np.float32) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    dirs = np.stack(
        [
            (xx - cx) / fx,
            (yy - cy) / fy,
            np.ones_like(xx),
        ],
        axis=-1,
    )

    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]
    rays_d = dirs @ rotation.T
    rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
    rays_o = np.broadcast_to(translation, rays_d.shape).copy()
    return rays_o.astype(np.float32), rays_d.astype(np.float32)


def latent_frame_groups(num_frames, temporal_downsample=4):
    latent_t = (num_frames - 1) // temporal_downsample + 1
    groups = [[0] * temporal_downsample]
    for i in range(1, latent_t):
        start = 1 + (i - 1) * temporal_downsample
        groups.append([min(start + j, num_frames - 1) for j in range(temporal_downsample)])
    return groups


def get_latent_chunk_with_pad(tensor, start_idx, num_frames, temporal_downsample=4):
    groups = latent_frame_groups(num_frames, temporal_downsample=temporal_downsample)
    chunks = []
    for group in groups:
        source_tensors = []
        for local_idx in group:
            frame_idx = min(start_idx + local_idx, tensor.shape[1] - 1)
            source_tensors.append(tensor[:, frame_idx])
        chunks.append(torch.cat(source_tensors, dim=0))
    return torch.stack(chunks, dim=1)


def build_episode_controls(ep_info, camera_names, view_height, view_width, args):
    total_frames = ep_info["T_ds"]
    abs_actions = ep_info["abs_actions"]

    control_frames_per_cam = {}
    ray_o_frames_per_cam = {}
    ray_d_frames_per_cam = {}

    for cam in camera_names:
        cam_info = ep_info["cameras"][cam]
        intrinsic = cam_info["intrinsic"].copy()
        orig_h = args.orig_height
        orig_w = args.orig_width
        if orig_h is None or orig_w is None:
            if ep_info["layout"] == "split":
                video_path = os.path.join(ep_info["video_dir"], f"{cam}_color.mp4")
                sample_frame = read_video_frame(video_path, int(ep_info["ds_indices"][0]))
            else:
                video_path = os.path.join(ep_info["chunk_paths"][0], f"{cam}_color.mp4")
                sample_frame = read_video_frame(video_path, 0)
            orig_h, orig_w = sample_frame.shape[:2]

        intrinsic[0, 0] *= view_width / orig_w
        intrinsic[0, 2] *= view_width / orig_w
        intrinsic[1, 1] *= view_height / orig_h
        intrinsic[1, 2] *= view_height / orig_h

        control_frames_per_cam[cam] = generate_traj_map(
            abs_actions,
            cam_info["w2c"],
            intrinsic,
            view_height,
            view_width,
            radius=args.traj_radius,
            radius_mode=args.traj_radius_mode,
            min_radius=args.traj_min_radius,
            max_radius=args.traj_max_radius,
            ref_depth=args.traj_ref_depth,
            near_depth=args.traj_near_depth,
            far_depth=args.traj_far_depth,
        )

        if args.output_raymap:
            ray_o_seq = []
            ray_d_seq = []
            if args.raymap_mode == "latent":
                if view_height % 8 != 0 or view_width % 8 != 0:
                    raise ValueError("--view_height and --view_width must be divisible by 8 for latent raymap.")
                latent_h = view_height // 8
                latent_w = view_width // 8
                latent_intrinsic = intrinsic.copy()
                latent_intrinsic[0, 0] *= latent_w / view_width
                latent_intrinsic[0, 2] *= latent_w / view_width
                latent_intrinsic[1, 1] *= latent_h / view_height
                latent_intrinsic[1, 2] *= latent_h / view_height
                for t in range(total_frames):
                    ray_o, ray_d = generate_raw_raymap(
                        intrinsic=latent_intrinsic,
                        c2w=cam_info["c2w"][t],
                        height=latent_h,
                        width=latent_w,
                    )
                    ray_o_seq.append(ray_o)
                    ray_d_seq.append(ray_d)
            else:
                for t in range(total_frames):
                    ray_o, ray_d = generate_raymap(
                        intrinsic=intrinsic,
                        c2w=cam_info["c2w"][t],
                        height=view_height,
                        width=view_width,
                        ray_o_vmin=args.ray_o_vmin,
                        ray_o_vmax=args.ray_o_vmax,
                        ray_d_vmin=args.ray_d_vmin,
                        ray_d_vmax=args.ray_d_vmax,
                    )
                    ray_o_seq.append(ray_o)
                    ray_d_seq.append(ray_d)
            ray_o_frames_per_cam[cam] = ray_o_seq
            ray_d_frames_per_cam[cam] = ray_d_seq

    control_video = []
    ray_map_o = []
    ray_map_d = []
    for t in range(total_frames):
        control_concat = np.concatenate([control_frames_per_cam[cam][t] for cam in camera_names], axis=0)
        control_video.append(Image.fromarray(control_concat))
        if args.output_raymap and args.raymap_mode == "image":
            ray_o_concat = np.concatenate([ray_o_frames_per_cam[cam][t] for cam in camera_names], axis=0)
            ray_d_concat = np.concatenate([ray_d_frames_per_cam[cam][t] for cam in camera_names], axis=0)
            ray_map_o.append(Image.fromarray(ray_o_concat))
            ray_map_d.append(Image.fromarray(ray_d_concat))

    if args.output_raymap and args.raymap_mode == "latent":
        ray_o_latents = []
        ray_d_latents = []
        for t in range(total_frames):
            ray_o_latents.append(np.concatenate([ray_o_frames_per_cam[cam][t] for cam in camera_names], axis=0))
            ray_d_latents.append(np.concatenate([ray_d_frames_per_cam[cam][t] for cam in camera_names], axis=0))
        ray_map_o = torch.from_numpy(np.stack(ray_o_latents, axis=0)).permute(3, 0, 1, 2).float()
        ray_map_d = torch.from_numpy(np.stack(ray_d_latents, axis=0)).permute(3, 0, 1, 2).float()

    return control_video, ray_map_o, ray_map_d


def get_chunk_with_pad(frames, start_idx, num_frames):
    chunk = frames[start_idx: start_idx + num_frames]
    if len(chunk) < num_frames:
        chunk = chunk + [frames[-1]] * (num_frames - len(chunk))
    return chunk


def load_pipe(base_model_paths, vace_model_path, device):
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    model_configs = [ModelConfig(path=path) for path in base_model_paths]
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
    )
    if vace_model_path:
        state_dict = load_state_dict(vace_model_path)
        if any(name.startswith("vace_global_") for name in state_dict):
            pipe.vace.enable_global_cross_attn(global_context_dim=16)
        if any(name.startswith("vace_ray_o_adapter.") or name.startswith("vace_ray_d_adapter.") for name in state_dict):
            pipe.vace.enable_latent_raymap_adapter()
        pipe.vace.load_state_dict(state_dict)
    return pipe


def save_frames(frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame.save(os.path.join(out_dir, f"frame_{idx:06d}.png"))


def run_episode(pipe, ep_info, args):
    camera_names = args.camera_names
    view_height = args.view_height
    view_width = args.view_width
    total_height = view_height * len(camera_names)

    first_frame = load_first_multiview_frame(ep_info, camera_names, view_height, view_width)
    control_video, ray_map_o, ray_map_d = build_episode_controls(ep_info, camera_names, view_height, view_width, args)

    total_frames = ep_info["T_ds"]
    if total_frames <= 0:
        raise RuntimeError(f"Episode {ep_info['episode_key']} is empty.")

    ep_out_dir = os.path.join(args.output_dir, ep_info["episode_key"])
    images_dir = os.path.join(ep_out_dir, "images")
    os.makedirs(ep_out_dir, exist_ok=True)

    if args.save_control_video:
        save_video(control_video, os.path.join(ep_out_dir, "control_video.mp4"), fps=args.fps, quality=5)
    if args.output_raymap and args.save_raymap_video and args.raymap_mode == "image":
        save_video(ray_map_o, os.path.join(ep_out_dir, "ray_map_o.mp4"), fps=args.fps, quality=5)
        save_video(ray_map_d, os.path.join(ep_out_dir, "ray_map_d.mp4"), fps=args.fps, quality=5)

    generated_frames = [first_frame]
    current_context = first_frame
    predict_frames = args.predict_frames
    fixed_num_frames = args.num_frames

    total_prediction_frames = max(total_frames - 1, 0)
    num_chunks = math.ceil(total_prediction_frames / predict_frames) if total_prediction_frames > 0 else 0

    for chunk_idx in range(num_chunks):
        context_idx = chunk_idx * predict_frames
        valid_prediction_count = min(predict_frames, total_prediction_frames - context_idx)
        print(
            f"[INFO] episode={ep_info['episode_key']} "
            f"chunk={chunk_idx + 1}/{num_chunks} "
            f"context_idx={context_idx} "
            f"valid_prediction_count={valid_prediction_count}"
        )

        control_chunk = get_chunk_with_pad(control_video, context_idx, fixed_num_frames)
        pipe_kwargs = {
            "prompt": args.prompt,
            "vace_video": control_chunk,
            "vace_reference_image": current_context,
            "height": total_height,
            "width": view_width,
            "num_frames": fixed_num_frames,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed + chunk_idx,
            "tiled": args.tiled,
        }
        if args.output_raymap:
            if args.raymap_mode == "latent":
                pipe_kwargs["ray_map_o"] = get_latent_chunk_with_pad(ray_map_o, context_idx, fixed_num_frames)
                pipe_kwargs["ray_map_d"] = get_latent_chunk_with_pad(ray_map_d, context_idx, fixed_num_frames)
            else:
                pipe_kwargs["ray_map_o"] = get_chunk_with_pad(ray_map_o, context_idx, fixed_num_frames)
                pipe_kwargs["ray_map_d"] = get_chunk_with_pad(ray_map_d, context_idx, fixed_num_frames)

        if args.dry_run:
            print(
                f"[DRY RUN] episode={ep_info['episode_key']} chunk={chunk_idx + 1}/{num_chunks} "
                f"context_idx={context_idx} valid_prediction_count={valid_prediction_count} "
                f"frame_size=({total_height}, {view_width}) "
                f"output_raymap={args.output_raymap} raymap_mode={args.raymap_mode}"
            )
            break

        generated = pipe(**pipe_kwargs)
        new_frames = generated[1: 1 + valid_prediction_count]
        for frame in new_frames:
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(np.array(frame))
            generated_frames.append(frame)
        current_context = generated_frames[-1]

    if not args.dry_run and len(generated_frames) != total_frames:
        raise RuntimeError(
            f"Episode {ep_info['episode_key']}: generated {len(generated_frames)} frames, expected {total_frames}."
        )

    save_frames(generated_frames, images_dir)
    save_video(generated_frames, os.path.join(ep_out_dir, "video.mp4"), fps=args.fps, quality=5)
    print(
        f"[INFO] episode={ep_info['episode_key']} "
        f"saved_frames={len(generated_frames)} output_dir={ep_out_dir}"
    )


def parse_devices(devices):
    if devices.strip().lower() == "auto":
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    out = []
    for item in devices.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(item if item.startswith("cuda") or item == "cpu" else f"cuda:{item}")
    return out


def split_evenly(items, num_buckets):
    buckets = [[] for _ in range(num_buckets)]
    for idx, item in enumerate(items):
        buckets[idx % num_buckets].append(item)
    return [bucket for bucket in buckets if bucket]


def run_episode_worker(worker_id, device, episodes, args_dict, downsample_step):
    args = argparse.Namespace(**args_dict)
    args.device = device
    print(
        f"[WORKER {worker_id}] device={device} episodes={len(episodes)} "
        f"first={episodes[0]['episode_key'] if episodes else 'none'}",
        flush=True,
    )
    pipe = None
    if not args.dry_run:
        if not torch.cuda.is_available() and device != "cpu":
            raise RuntimeError("CUDA is required unless --devices cpu is used.")
        pipe = load_pipe(args.base_model_paths, args.vace_model_path, device)
    for ep in episodes:
        ep_info = build_episode_info(ep, downsample_step, args.camera_names)
        run_episode(pipe, ep_info, args)
    print(f"[WORKER {worker_id}] done", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", type=str, default="/mnt/workspace/zsq/Agi2024subset_split/val")
    parser.add_argument("--output_dir", type=str, default="/mnt/workspace/zsq/Agi2024subset_split/val_gen_multiview_ada")
    parser.add_argument("--prompt", type=str, default="机械臂按照要求移动夹爪执行任务")
    parser.add_argument("--camera_names", nargs="+", default=["head", "hand_left", "hand_right"])
    parser.add_argument("--original_hz", type=int, default=30)
    parser.add_argument("--target_hz", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=9)
    parser.add_argument("--predict_frames", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--view_height", type=int, default=320)
    parser.add_argument("--view_width", type=int, default=512)
    parser.add_argument("--orig_height", type=int, default=None)
    parser.add_argument("--orig_width", type=int, default=None)
    parser.add_argument("--traj_radius", type=int, default=50)
    parser.add_argument("--traj_radius_mode", type=str, default="perspective", choices=["constant", "perspective", "depth_norm"])
    parser.add_argument("--traj_min_radius", type=int, default=20)
    parser.add_argument("--traj_max_radius", type=int, default=60)
    parser.add_argument("--traj_ref_depth", type=float, default=0.30)
    parser.add_argument("--traj_near_depth", type=float, default=0.19)
    parser.add_argument("--traj_far_depth", type=float, default=0.69)
    parser.add_argument("--output_raymap", action="store_true", default=True)
    parser.add_argument("--no_output_raymap", dest="output_raymap", action="store_false")
    parser.add_argument("--raymap_mode", type=str, default="latent", choices=["image", "latent"])
    parser.add_argument("--ray_o_vmin", type=float, default=-1.5)
    parser.add_argument("--ray_o_vmax", type=float, default=1.5)
    parser.add_argument("--ray_d_vmin", type=float, default=-1.0)
    parser.add_argument("--ray_d_vmax", type=float, default=1.0)
    parser.add_argument("--episode_limit", type=int, default=None)
    parser.add_argument("--episode_filter", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--devices", type=str, default="", help="Comma-separated devices for episode-level parallelism, e.g. 0,1,2,3 or cuda:0,cuda:1. Empty means use --device.")
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--save_control_video", action="store_true")
    parser.add_argument("--save_raymap_video", action="store_true")
    parser.add_argument(
        "--base_model_paths",
        nargs="+",
        default=[
            "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
            "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/models_t5_umt5-xxl-enc-bf16.pth",
            "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/Wan2.1_VAE.pth",
        ],
    )
    parser.add_argument(
        "--vace_model_path",
        type=str,
        default="/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.1-VACE-1.3B-multiview-raymap-perspective-latentraymap-adapter/epoch-49.safetensors",
    )
    args = parser.parse_args()

    if args.num_frames != args.predict_frames + 1:
        raise ValueError("--num_frames must equal --predict_frames + 1 for the current autoregressive setup.")
    if args.original_hz % args.target_hz != 0:
        raise ValueError("--original_hz must be divisible by --target_hz.")

    downsample_step = args.original_hz // args.target_hz
    episodes = discover_episodes(args.val_path)
    print(f"[INFO] Found {len(episodes)} episode(s) under {args.val_path}")

    if args.episode_filter is not None:
        episodes = [ep for ep in episodes if ep["episode_key"] == args.episode_filter]
    elif args.episode_limit is not None:
        episodes = episodes[: args.episode_limit]
    print(f"[INFO] Will process {len(episodes)} episode(s)")

    if not episodes:
        raise RuntimeError("No episodes selected.")

    raymap_tag = f"{args.raymap_mode}raymap" if args.output_raymap else "noraymap"
    args.output_dir = f"{args.output_dir}_{args.traj_radius_mode}_{raymap_tag}"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {args.output_dir}")

    devices = parse_devices(args.devices) if args.devices else [args.device]
    worker_count = min(max(1, args.max_workers), len(devices), len(episodes))

    if worker_count <= 1:
        if args.dry_run:
            pipe = None
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for this script.")
            pipe = load_pipe(args.base_model_paths, args.vace_model_path, args.device)

        for ep in episodes:
            ep_info = build_episode_info(ep, downsample_step, args.camera_names)
            run_episode(pipe, ep_info, args)
    else:
        if args.dry_run:
            print("[INFO] dry_run with multi-worker assignment enabled; no model will be loaded.")
        print(f"[INFO] Episode-level parallelism: workers={worker_count}, devices={devices[:worker_count]}")
        buckets = split_evenly(episodes, worker_count)
        ctx = mp.get_context("spawn")
        procs = []
        args_dict = vars(args).copy()
        for worker_id, bucket in enumerate(buckets):
            p = ctx.Process(
                target=run_episode_worker,
                args=(worker_id, devices[worker_id], bucket, args_dict, downsample_step),
            )
            p.start()
            procs.append(p)
        failed = []
        for p in procs:
            p.join()
            if p.exitcode != 0:
                failed.append(p.exitcode)
        if failed:
            raise RuntimeError(f"{len(failed)} worker(s) failed with exit codes: {failed}")

    print(f"[DONE] Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
