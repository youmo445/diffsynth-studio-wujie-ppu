import argparse
import json
import math
import os
import sys
import time

WUJIE_REPO = "/data/zsq/diffsynth-studio-wujie-ppu"
if WUJIE_REPO not in sys.path:
    sys.path.insert(0, WUJIE_REPO)

import h5py
import imageio.v3 as iio
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image

from diffsynth import load_state_dict
from diffsynth.trainers.utils_codex import (
    apply_eef2cam_visual_rotation,
    generate_traj_map,
)


def discover_episodes(base_path):
    proprio_base = os.path.join(base_path, "proprio_stats")
    if not os.path.isdir(proprio_base):
        raise RuntimeError(f"Only split layout is supported in this benchmark: {base_path}")

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
                    "h5_path": h5_path,
                    "video_dir": os.path.join(base_path, "observations", task, ep, "videos"),
                    "camera_dir": os.path.join(base_path, "parameters", task, ep, "parameters", "camera"),
                }
            )
    if not episodes:
        raise RuntimeError(f"No episodes found under {base_path}")
    return episodes


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
        "video_dir": ep["video_dir"],
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
    orig_idx = int(ep_info["ds_indices"][0])
    parts = []
    for cam in camera_names:
        video_path = os.path.join(ep_info["video_dir"], f"{cam}_color.mp4")
        frame = read_video_frame(video_path, orig_idx)
        parts.append(np.array(Image.fromarray(frame).resize((view_width, view_height), Image.BILINEAR)))
    return Image.fromarray(np.concatenate(parts, axis=0))


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


def build_episode_controls(ep_info, args):
    total_frames = ep_info["T_ds"]
    abs_actions = ep_info["abs_actions"]
    camera_names = args.camera_names

    control_frames_per_cam = {}
    ray_o_frames_per_cam = {}
    ray_d_frames_per_cam = {}

    for cam in camera_names:
        cam_info = ep_info["cameras"][cam]
        intrinsic = cam_info["intrinsic"].copy()

        if args.orig_height is None or args.orig_width is None:
            sample_path = os.path.join(ep_info["video_dir"], f"{cam}_color.mp4")
            sample_frame = read_video_frame(sample_path, int(ep_info["ds_indices"][0]))
            orig_h, orig_w = sample_frame.shape[:2]
        else:
            orig_h, orig_w = args.orig_height, args.orig_width

        intrinsic[0, 0] *= args.view_width / orig_w
        intrinsic[0, 2] *= args.view_width / orig_w
        intrinsic[1, 1] *= args.view_height / orig_h
        intrinsic[1, 2] *= args.view_height / orig_h

        control_frames_per_cam[cam] = generate_traj_map(
            abs_actions,
            cam_info["w2c"],
            intrinsic,
            args.view_height,
            args.view_width,
            radius=args.traj_radius,
            radius_mode="perspective",
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
                latent_h = args.view_height // 8
                latent_w = args.view_width // 8
                latent_intrinsic = intrinsic.copy()
                latent_intrinsic[0, 0] *= latent_w / args.view_width
                latent_intrinsic[0, 2] *= latent_w / args.view_width
                latent_intrinsic[1, 1] *= latent_h / args.view_height
                latent_intrinsic[1, 2] *= latent_h / args.view_height
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
                        height=args.view_height,
                        width=args.view_width,
                        ray_o_vmin=np.array(args.ray_o_vmin, dtype=np.float32),
                        ray_o_vmax=np.array(args.ray_o_vmax, dtype=np.float32),
                        ray_d_vmin=np.array(args.ray_d_vmin, dtype=np.float32),
                        ray_d_vmax=np.array(args.ray_d_vmax, dtype=np.float32),
                    )
                    ray_o_seq.append(ray_o)
                    ray_d_seq.append(ray_d)
            ray_o_frames_per_cam[cam] = ray_o_seq
            ray_d_frames_per_cam[cam] = ray_d_seq

    control_video = []
    ray_map_o = []
    ray_map_d = []
    for t in range(total_frames):
        control_video.append(
            Image.fromarray(np.concatenate([control_frames_per_cam[cam][t] for cam in camera_names], axis=0))
        )
        if args.output_raymap and args.raymap_mode == "image":
            ray_map_o.append(
                Image.fromarray(np.concatenate([ray_o_frames_per_cam[cam][t] for cam in camera_names], axis=0))
            )
            ray_map_d.append(
                Image.fromarray(np.concatenate([ray_d_frames_per_cam[cam][t] for cam in camera_names], axis=0))
            )
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
        pipe.vace.load_state_dict(state_dict)
    return pipe


def benchmark_episode(pipe, ep_info, args):
    first_frame = load_first_multiview_frame(
        ep_info, args.camera_names, args.view_height, args.view_width
    )
    control_video, ray_map_o, ray_map_d = build_episode_controls(ep_info, args)

    total_frames = ep_info["T_ds"]
    num_chunks_all = max(1, int(math.ceil(max(total_frames - 1, 0) / float(args.predict_frames))))
    num_chunks = min(args.num_chunks, num_chunks_all)
    total_height = args.view_height * len(args.camera_names)

    current_context = first_frame
    measured_times = []
    measured_frames = 0
    records = []

    for chunk_idx in range(num_chunks):
        context_idx = chunk_idx * args.predict_frames
        remaining_predictions = max(total_frames - 1 - context_idx, 0)
        valid_prediction_count = min(args.predict_frames, remaining_predictions)

        if valid_prediction_count <= 0:
            break

        pipe_kwargs = {
            "prompt": args.prompt,
            "vace_video": get_chunk_with_pad(control_video, context_idx, args.num_frames),
            "vace_reference_image": current_context,
            "height": total_height,
            "width": args.view_width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed + chunk_idx,
            "tiled": False,
        }
        if args.output_raymap:
            pipe_kwargs["ray_map_o"] = (
                get_latent_chunk_with_pad(ray_map_o, context_idx, args.num_frames)
                if args.raymap_mode == "latent"
                else get_chunk_with_pad(ray_map_o, context_idx, args.num_frames)
            )
            pipe_kwargs["ray_map_d"] = (
                get_latent_chunk_with_pad(ray_map_d, context_idx, args.num_frames)
                if args.raymap_mode == "latent"
                else get_chunk_with_pad(ray_map_d, context_idx, args.num_frames)
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        generated = pipe(**pipe_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        new_frames = generated[1: 1 + valid_prediction_count]
        current_context = new_frames[-1] if len(new_frames) > 0 else current_context

        chunk_fps = valid_prediction_count / max(dt, 1e-8)
        is_warmup = chunk_idx < args.warmup_chunks
        if not is_warmup:
            measured_times.append(dt)
            measured_frames += valid_prediction_count

        record = {
            "chunk_idx": chunk_idx + 1,
            "valid_prediction_count": valid_prediction_count,
            "seconds": dt,
            "fps": chunk_fps,
            "warmup": is_warmup,
        }
        records.append(record)
        print(
            f"[FPS] episode={ep_info['episode_key']} "
            f"chunk={chunk_idx + 1}/{num_chunks} "
            f"pred_frames={valid_prediction_count} "
            f"time={dt:.4f}s fps={chunk_fps:.3f} "
            f"warmup={is_warmup}",
            flush=True,
        )

    measured_total_time = float(sum(measured_times))
    overall_fps = measured_frames / max(measured_total_time, 1e-8)
    result = {
        "episode_key": ep_info["episode_key"],
        "num_chunks_requested": args.num_chunks,
        "num_chunks_run": len(records),
        "warmup_chunks": args.warmup_chunks,
        "measured_chunks": sum(not r["warmup"] for r in records),
        "measured_frames": measured_frames,
        "measured_total_time_sec": measured_total_time,
        "overall_fps": overall_fps,
        "records": records,
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", type=str, default="/data/zsq/agibot2024/Agi2024subset_split/val")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=5)
    parser.add_argument("--warmup_chunks", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--camera_names", nargs="+", default=["head", "hand_left", "hand_right"])
    parser.add_argument("--orig_height", type=int, default=480)
    parser.add_argument("--orig_width", type=int, default=640)
    parser.add_argument("--view_height", type=int, default=320)
    parser.add_argument("--view_width", type=int, default=512)
    parser.add_argument("--original_hz", type=int, default=30)
    parser.add_argument("--target_hz", type=int, default=5)
    parser.add_argument("--predict_frames", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--traj_radius", type=int, default=3)
    parser.add_argument("--traj_min_radius", type=float, default=2.0)
    parser.add_argument("--traj_max_radius", type=float, default=7.0)
    parser.add_argument("--traj_ref_depth", type=float, default=1.0)
    parser.add_argument("--traj_near_depth", type=float, default=0.3)
    parser.add_argument("--traj_far_depth", type=float, default=2.0)
    parser.add_argument("--output_raymap", action="store_true", default=True)
    parser.add_argument("--no_output_raymap", dest="output_raymap", action="store_false")
    parser.add_argument("--raymap_mode", type=str, default="image", choices=["image", "latent"])
    parser.add_argument("--ray_o_vmin", type=float, nargs=3, default=[-1.5, -1.5, -1.5])
    parser.add_argument("--ray_o_vmax", type=float, nargs=3, default=[1.5, 1.5, 1.5])
    parser.add_argument("--ray_d_vmin", type=float, nargs=3, default=[-1.0, -1.0, -1.0])
    parser.add_argument("--ray_d_vmax", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument(
        "--base_model_paths",
        nargs="+",
        default=[
            "/data/zsq/Wan2.1-1.3B-VACE/diffusion_pytorch_model.safetensors",
            "/data/zsq/Wan2.1-1.3B-VACE/models_t5_umt5-xxl-enc-bf16.pth",
            "/data/zsq/Wan2.1-1.3B-VACE/Wan2.1_VAE.pth",
        ],
    )
    parser.add_argument(
        "--vace_model_path",
        type=str,
        default="/data/zsq/diffsynth-studio-wujie-ppu/outputs/Wan2.1-VACE-1.3B-multiview-constant-noraymap/epoch-79.safetensors",
    )
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()
    if args.original_hz % args.target_hz != 0:
        raise ValueError("--original_hz must be divisible by --target_hz.")
    if args.num_frames != args.predict_frames + 1:
        raise ValueError("--num_frames must equal --predict_frames + 1.")
    args.downsample_step = args.original_hz // args.target_hz
    return args


def main():
    args = parse_args()
    episodes = discover_episodes(args.val_path)
    if not (0 <= args.episode_index < len(episodes)):
        raise IndexError(f"--episode_index out of range: {args.episode_index}, total episodes={len(episodes)}")

    ep = episodes[args.episode_index]
    ep_info = build_episode_info(ep, args.downsample_step, args.camera_names)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    pipe = load_pipe(args.base_model_paths, args.vace_model_path, args.device)
    result = benchmark_episode(pipe, ep_info, args)

    print(
        f"[SUMMARY] episode={result['episode_key']} "
        f"measured_chunks={result['measured_chunks']} "
        f"measured_frames={result['measured_frames']} "
        f"time={result['measured_total_time_sec']:.4f}s "
        f"overall_fps={result['overall_fps']:.3f}",
        flush=True,
    )

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[SUMMARY] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
