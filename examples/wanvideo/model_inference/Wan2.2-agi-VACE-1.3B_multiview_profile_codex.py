"""
Detailed per-chunk profiler for multiview Wan VACE inference.

Fixed setting by default:
- three-view concatenated input
- traj_radius_mode = perspective
- output_raymap = True

This script focuses on timing:
- whole chunk generation
- DiT/model_fn total time
- each VAE encode/decode stage, with semantic labels
"""

import argparse
import json
import math
import os
import time
import types
from collections import defaultdict

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
        raise RuntimeError(f"Only split layout is supported in this profiler: {base_path}")

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

        ray_o_seq = []
        ray_d_seq = []
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
        ray_map_o.append(
            Image.fromarray(np.concatenate([ray_o_frames_per_cam[cam][t] for cam in camera_names], axis=0))
        )
        ray_map_d.append(
            Image.fromarray(np.concatenate([ray_d_frames_per_cam[cam][t] for cam in camera_names], axis=0))
        )
    return control_video, ray_map_o, ray_map_d


def get_chunk_with_pad(frames, start_idx, num_frames):
    chunk = frames[start_idx: start_idx + num_frames]
    if len(chunk) < num_frames:
        chunk = chunk + [frames[-1]] * (num_frames - len(chunk))
    return chunk


class ChunkProfiler:
    def __init__(self):
        self.current = None
        self.chunk_idx = None

    def start_chunk(self, chunk_idx):
        self.chunk_idx = chunk_idx
        self.current = {
            "chunk_idx": chunk_idx,
            "timings": defaultdict(float),
            "calls": defaultdict(int),
        }

    def add(self, label, seconds):
        self.current["timings"][label] += float(seconds)
        self.current["calls"][label] += 1

    def finish_chunk(self):
        out = {
            "chunk_idx": self.current["chunk_idx"],
            "timings": dict(self.current["timings"]),
            "calls": dict(self.current["calls"]),
        }
        self.current = None
        self.chunk_idx = None
        return out


def install_profilers(pipe, profiler):
    original_model_fn = pipe.model_fn

    def timed_model_fn(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = original_model_fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        profiler.add("dit_model_fn_total", time.perf_counter() - t0)
        return out

    pipe.model_fn = timed_model_fn

    original_decode = pipe.vae.decode

    def timed_decode(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = original_decode(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        profiler.add("vae_decode_output_video", time.perf_counter() - t0)
        return out

    pipe.vae.decode = timed_decode

    target_unit = None
    for unit in pipe.units:
        if unit.__class__.__name__ == "WanVideoUnit_VACE_raymap":
            target_unit = unit
            break
    if target_unit is None:
        raise RuntimeError("Cannot find WanVideoUnit_VACE_raymap in pipe.units")

    def timed_vace_process(
        self,
        pipe,
        vace_video,
        vace_video_mask,
        vace_reference_image,
        ray_map_o,
        ray_map_d,
        vace_scale,
        height,
        width,
        num_frames,
        tiled,
        tile_size,
        tile_stride,
    ):
        if ray_map_o is None and ray_map_d is None:
            raise RuntimeError("This profiler expects ray_map_o and ray_map_d to both be provided.")
        if ray_map_o is None or ray_map_d is None:
            raise ValueError("ray_map_o and ray_map_d must be provided together.")
        if vace_video is None and vace_video_mask is None and vace_reference_image is None:
            return {"vace_context": None, "vace_scale": vace_scale}

        pipe.load_models_to_device(["vae"])
        if vace_video is None:
            vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            vace_video = pipe.preprocess_video(vace_video)

        if vace_video_mask is None:
            vace_video_mask = torch.ones_like(vace_video)
        else:
            vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)

        inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
        reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        inactive = pipe.vae.encode(
            inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        profiler.add("vae_encode_inactive_vace_video", time.perf_counter() - t0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        reactive = pipe.vae.encode(
            reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        profiler.add("vae_encode_reactive_vace_video", time.perf_counter() - t0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ray_map_o_latents = self._encode_video_latents(pipe, ray_map_o, tiled, tile_size, tile_stride)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        profiler.add("vae_encode_ray_map_o", time.perf_counter() - t0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ray_map_d_latents = self._encode_video_latents(pipe, ray_map_d, tiled, tile_size, tile_stride)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        profiler.add("vae_encode_ray_map_d", time.perf_counter() - t0)

        vace_video_latents = torch.concat((inactive, reactive, ray_map_o_latents, ray_map_d_latents), dim=1)
        vace_mask_latents = self._mask_latents(vace_video_mask, mask_channels=32)

        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            bs, c, f, h, w = vace_reference_image.shape
            new_vace_ref_images = []
            for j in range(f):
                new_vace_ref_images.append(vace_reference_image[0, :, j:j + 1])

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            vace_reference_latents = pipe.vae.encode(
                new_vace_ref_images,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            profiler.add("vae_encode_vace_reference_image", time.perf_counter() - t0)

            vace_reference_latents = torch.concat(
                (vace_reference_latents, torch.zeros_like(vace_reference_latents).repeat(1, 3, 1, 1, 1)),
                dim=1,
            )
            vace_reference_latents = [u.unsqueeze(0) for u in vace_reference_latents]
            vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
            vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents), dim=2)

        vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
        return {"vace_context": vace_context, "vace_scale": vace_scale}

    target_unit.process = types.MethodType(timed_vace_process, target_unit)


def load_pipe(base_model_paths, vace_model_path, device, profiler):
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    model_configs = [ModelConfig(path=path) for path in base_model_paths]
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
    )
    if vace_model_path:
        state_dict = load_state_dict(vace_model_path)
        pipe.vace.load_state_dict(state_dict)
    install_profilers(pipe, profiler)
    return pipe


def benchmark_episode(pipe, profiler, ep_info, args):
    first_frame = load_first_multiview_frame(ep_info, args.camera_names, args.view_height, args.view_width)
    control_video, ray_map_o, ray_map_d = build_episode_controls(ep_info, args)

    total_frames = ep_info["T_ds"]
    num_chunks_all = max(1, int(math.ceil(max(total_frames - 1, 0) / float(args.predict_frames))))
    num_chunks = min(args.num_chunks, num_chunks_all)
    total_height = args.view_height * len(args.camera_names)
    current_context = first_frame
    results = []

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
            "ray_map_o": get_chunk_with_pad(ray_map_o, context_idx, args.num_frames),
            "ray_map_d": get_chunk_with_pad(ray_map_d, context_idx, args.num_frames),
            "height": total_height,
            "width": args.view_width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed + chunk_idx,
            "tiled": False,
        }

        profiler.start_chunk(chunk_idx + 1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        generated = pipe(**pipe_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_sec = time.perf_counter() - t0

        new_frames = generated[1: 1 + valid_prediction_count]
        current_context = new_frames[-1] if len(new_frames) > 0 else current_context

        chunk_result = profiler.finish_chunk()
        chunk_result["pipe_total_sec"] = total_sec
        chunk_result["predicted_frames"] = valid_prediction_count
        chunk_result["fps"] = valid_prediction_count / max(total_sec, 1e-8)
        results.append(chunk_result)

        timings = chunk_result["timings"]
        calls = chunk_result["calls"]
        named_order = [
            ("vae_encode_inactive_vace_video", "VAE encode inactive branch (masked-out action/ray video)"),
            ("vae_encode_reactive_vace_video", "VAE encode reactive branch (active action/ray video)"),
            ("vae_encode_ray_map_o", "VAE encode ray_map_o (camera-origin video)"),
            ("vae_encode_ray_map_d", "VAE encode ray_map_d (camera-direction video)"),
            ("vae_encode_vace_reference_image", "VAE encode vace_reference_image (current context frame)"),
            ("dit_model_fn_total", "DiT / model_fn total denoising time"),
            ("vae_decode_output_video", "VAE decode final latent video"),
        ]

        print(
            f"[PROFILE] episode={ep_info['episode_key']} chunk={chunk_idx + 1}/{num_chunks} "
            f"pred_frames={valid_prediction_count} pipe_total={total_sec:.4f}s fps={chunk_result['fps']:.3f}",
            flush=True,
        )
        for key, desc in named_order:
            if key in timings:
                print(
                    f"  - {key}: {timings[key]:.4f}s "
                    f"(calls={calls.get(key, 0)}) | {desc}",
                    flush=True,
                )
        known = sum(timings.get(k, 0.0) for k, _ in named_order)
        other = max(total_sec - known, 0.0)
        print(f"  - other_overhead: {other:.4f}s", flush=True)

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", type=str, default="/mnt/workspace/zsq/Agi2024subset_split/val")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=3)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--camera_names", nargs="+", default=["head", "hand_left", "hand_right"])
    parser.add_argument("--orig_height", type=int, default=480)
    parser.add_argument("--orig_width", type=int, default=640)
    parser.add_argument("--view_height", type=int, default=480)
    parser.add_argument("--view_width", type=int, default=640)
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
    parser.add_argument("--ray_o_vmin", type=float, nargs=3, default=[-1.5, -1.5, -1.5])
    parser.add_argument("--ray_o_vmax", type=float, nargs=3, default=[1.5, 1.5, 1.5])
    parser.add_argument("--ray_d_vmin", type=float, nargs=3, default=[-1.0, -1.0, -1.0])
    parser.add_argument("--ray_d_vmax", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument(
        "--base_model_paths",
        nargs="+",
        default=[
            "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
            "/mnt/workspace/zsq/Wan_model/Wan2.2-Fun-A14B-Control/models_t5_umt5-xxl-enc-bf16.pth",
            "/mnt/workspace/zsq/Wan_model/Wan2.2-Fun-A14B-Control/Wan2.1_VAE.pth",
        ],
    )
    parser.add_argument(
        "--vace_model_path",
        type=str,
        default="/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.1-VACE-1.3B-multiview-perspective/epoch-9.safetensors",
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiler.")

    ep_info = build_episode_info(episodes[args.episode_index], args.downsample_step, args.camera_names)
    profiler = ChunkProfiler()
    pipe = load_pipe(args.base_model_paths, args.vace_model_path, args.device, profiler)
    results = benchmark_episode(pipe, profiler, ep_info, args)

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[PROFILE] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
