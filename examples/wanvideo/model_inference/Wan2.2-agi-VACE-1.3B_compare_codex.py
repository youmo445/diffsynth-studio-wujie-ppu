"""
Build a comparison video for one multiview trajectory:
GT | generated | action_map | optional ray_map_o | optional ray_map_d

The generated column is produced on the fly from GT-derived controls instead of
reading pre-generated frames from disk.
"""

import argparse
import json
import os

import h5py
import imageio
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
    episodes = {}
    if os.path.isdir(proprio_base):
        for task in sorted(os.listdir(proprio_base)):
            task_path = os.path.join(proprio_base, task)
            if not os.path.isdir(task_path):
                continue
            for ep in sorted(os.listdir(task_path)):
                h5_path = os.path.join(task_path, ep, "proprio_stats.h5")
                if not os.path.exists(h5_path):
                    continue
                key = f"{task}-{ep}"
                episodes[key] = {
                    "episode_key": key,
                    "h5_path": h5_path,
                    "video_dir": os.path.join(base_path, "observations", task, ep, "videos"),
                    "camera_dir": os.path.join(base_path, "parameters", task, ep, "parameters", "camera"),
                }
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


def load_video_frames_batch(video_path, frame_indices):
    frame_indices = [int(i) for i in frame_indices]
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        frames = vr.get_batch(frame_indices).asnumpy()
        del vr
        if frames.shape[-1] == 4:
            frames = frames[..., :3]
        return frames
    except Exception:
        frames = []
        for idx in frame_indices:
            frame = iio.imread(video_path, index=idx)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            frames.append(frame)
        return np.stack(frames, axis=0)


def load_gt_frames_range(ep_info, camera_names, frame_slice, view_height, view_width):
    raw_indices = ep_info["ds_indices"][frame_slice]
    raw_hw_by_cam = {}
    resized_by_cam = {}
    for cam in camera_names:
        video_path = os.path.join(ep_info["video_dir"], f"{cam}_color.mp4")
        frames = load_video_frames_batch(video_path, raw_indices.tolist())
        raw_hw_by_cam[cam] = tuple(frames[0].shape[:2])
        resized_by_cam[cam] = [
            np.array(Image.fromarray(frame).resize((view_width, view_height), Image.BILINEAR))
            for frame in frames
        ]

    gt_frames = []
    num_frames = len(raw_indices)
    for i in range(num_frames):
        parts = [resized_by_cam[cam][i] for cam in camera_names]
        gt_frames.append(Image.fromarray(np.concatenate(parts, axis=0)))
    return gt_frames, raw_hw_by_cam


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


def perturb_abs_actions(abs_actions, args):
    out = abs_actions.copy()
    out = apply_gripper_perturb(out, args)

    if args.eef_perturb_mode == "none":
        return out

    total_frames = out.shape[0]
    start_idx = max(0, min(args.eef_perturb_start, total_frames - 1))
    if args.eef_perturb_length is None:
        end_idx = total_frames
    else:
        end_idx = min(total_frames, start_idx + max(args.eef_perturb_length, 0))
    if end_idx <= start_idx:
        return out

    if args.eef_arm in ("left", "both"):
        arm_slices = [(slice(0, 3), slice(3, 7))]
    else:
        arm_slices = []
    if args.eef_arm in ("right", "both"):
        arm_slices.append((slice(8, 11), slice(11, 15)))

    if args.eef_perturb_mode == "hold":
        for pos_slice, quat_slice in arm_slices:
            ref_pos = out[start_idx, pos_slice].copy()
            ref_quat = out[start_idx, quat_slice].copy()
            out[start_idx:end_idx, pos_slice] = ref_pos
            out[start_idx:end_idx, quat_slice] = ref_quat
        return out

    if args.eef_perturb_mode == "jitter":
        rng = np.random.default_rng(args.eef_perturb_seed)
        for pos_slice, quat_slice in arm_slices:
            pos_noise = rng.normal(
                loc=0.0,
                scale=args.eef_jitter_pos_std,
                size=(end_idx - start_idx, pos_slice.stop - pos_slice.start),
            ).astype(np.float32)
            quat_noise = rng.normal(
                loc=0.0,
                scale=args.eef_jitter_quat_std,
                size=(end_idx - start_idx, quat_slice.stop - quat_slice.start),
            ).astype(np.float32)
            out[start_idx:end_idx, pos_slice] += pos_noise
            out[start_idx:end_idx, quat_slice] += quat_noise
            quat = out[start_idx:end_idx, quat_slice]
            quat /= np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8
            out[start_idx:end_idx, quat_slice] = quat
        return out

    raise ValueError(f"Unsupported eef_perturb_mode: {args.eef_perturb_mode}")


def apply_gripper_perturb(abs_actions, args):
    if args.gripper_perturb_mode == "none":
        return abs_actions

    out = abs_actions.copy()
    total_frames = out.shape[0]
    if total_frames == 0:
        return out

    start_idx = max(0, min(args.gripper_perturb_start, total_frames - 1))
    if args.gripper_perturb_length is None:
        end_idx = total_frames
    else:
        end_idx = min(total_frames, start_idx + max(args.gripper_perturb_length, 0))
    if end_idx <= start_idx:
        return out

    value_map = {
        "open": float(args.gripper_open_value),
        "close": float(args.gripper_close_value),
    }
    arm_to_index = {"left": 7, "right": 15}
    arm_modes = {
        "left": args.left_gripper_state,
        "right": args.right_gripper_state,
    }

    if args.gripper_perturb_mode == "hold":
        for arm, grip_idx in arm_to_index.items():
            if arm_modes[arm] != "keep":
                out[start_idx:end_idx, grip_idx] = value_map[arm_modes[arm]]
        return out

    raise ValueError(f"Unsupported gripper_perturb_mode: {args.gripper_perturb_mode}")


def build_controls_range(ep_info, camera_names, frame_slice, raw_hw_by_cam, view_height, view_width, args):
    abs_actions = ep_info["abs_actions"][frame_slice]
    control_abs_actions = perturb_abs_actions(abs_actions, args)
    total_frames = control_abs_actions.shape[0]
    action_map_per_cam = {}
    ray_o_per_cam = {}
    ray_d_per_cam = {}

    for cam in camera_names:
        cam_info = ep_info["cameras"][cam]
        orig_h, orig_w = raw_hw_by_cam[cam]
        intrinsic = cam_info["intrinsic"].copy()
        intrinsic[0, 0] *= view_width / orig_w
        intrinsic[0, 2] *= view_width / orig_w
        intrinsic[1, 1] *= view_height / orig_h
        intrinsic[1, 2] *= view_height / orig_h

        action_map_per_cam[cam] = generate_traj_map(
            control_abs_actions,
            cam_info["w2c"][frame_slice],
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
            for t in range(total_frames):
                ray_o, ray_d = generate_raymap(
                    intrinsic=intrinsic,
                    c2w=cam_info["c2w"][frame_slice][t],
                    height=view_height,
                    width=view_width,
                    ray_o_vmin=args.ray_o_vmin,
                    ray_o_vmax=args.ray_o_vmax,
                    ray_d_vmin=args.ray_d_vmin,
                    ray_d_vmax=args.ray_d_vmax,
                )
                ray_o_seq.append(ray_o)
                ray_d_seq.append(ray_d)
            ray_o_per_cam[cam] = ray_o_seq
            ray_d_per_cam[cam] = ray_d_seq

    action_maps = []
    ray_map_o = []
    ray_map_d = []
    for t in range(total_frames):
        action_maps.append(Image.fromarray(np.concatenate([action_map_per_cam[cam][t] for cam in camera_names], axis=0)))
        if args.output_raymap:
            ray_map_o.append(Image.fromarray(np.concatenate([ray_o_per_cam[cam][t] for cam in camera_names], axis=0)))
            ray_map_d.append(Image.fromarray(np.concatenate([ray_d_per_cam[cam][t] for cam in camera_names], axis=0)))
    return action_maps, ray_map_o, ray_map_d


def load_pipe(base_model_paths, vace_model_path, device):
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    model_configs = [ModelConfig(path=path) for path in base_model_paths]
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
    )
    state_dict = load_state_dict(vace_model_path)
    pipe.vace.load_state_dict(state_dict)
    return pipe


def get_chunk_with_pad(frames, start_idx, num_frames):
    chunk = frames[start_idx: start_idx + num_frames]
    if len(chunk) < num_frames:
        chunk = chunk + [frames[-1]] * (num_frames - len(chunk))
    return chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", type=str, default="/mnt/workspace/zsq/Agi2024subset_split/val")
    parser.add_argument("--output_dir", type=str, default="/mnt/workspace/zsq/DiffSynth-Studio/outputs/compare_multiview_codex")
    parser.add_argument("--episode_key", type=str, default="357-712687")
    parser.add_argument("--camera_names", nargs="+", default=["head", "hand_left", "hand_right"])
    parser.add_argument("--original_hz", type=int, default=30)
    parser.add_argument("--target_hz", type=int, default=5)
    parser.add_argument("--view_height", type=int, default=320)
    parser.add_argument("--view_width", type=int, default=512)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--num_frames", type=int, default=9)
    parser.add_argument("--predict_frames", type=int, default=8)
    parser.add_argument("--start_chunk", type=int, default=15, help="1-based chunk index to start from")
    parser.add_argument("--num_chunks", type=int, default=60, help="How many chunks to visualize")
    parser.add_argument("--traj_radius", type=int, default=50)
    parser.add_argument("--traj_radius_mode", type=str, default="perspective", choices=["constant", "perspective", "depth_norm"])
    parser.add_argument("--traj_min_radius", type=int, default=20)
    parser.add_argument("--traj_max_radius", type=int, default=60)
    parser.add_argument("--traj_ref_depth", type=float, default=0.30)
    parser.add_argument("--traj_near_depth", type=float, default=0.19)
    parser.add_argument("--traj_far_depth", type=float, default=0.69)
    parser.add_argument("--output_raymap", action="store_true")
    parser.add_argument("--ray_o_vmin", type=float, default=-1.5)
    parser.add_argument("--ray_o_vmax", type=float, default=1.5)
    parser.add_argument("--ray_d_vmin", type=float, default=-1.0)
    parser.add_argument("--ray_d_vmax", type=float, default=1.0)
    parser.add_argument("--output_name", type=str, default="comparison.mp4")
    parser.add_argument("--prompt", type=str, default="机械臂按照要求移动夹爪执行任务")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eef_perturb_mode", type=str, default="none", choices=["none", "hold", "jitter"])
    parser.add_argument("--eef_arm", type=str, default="left", choices=["left", "right", "both"])
    parser.add_argument("--eef_perturb_start", type=int, default=0, help="0-based local frame index within the selected range")
    parser.add_argument("--eef_perturb_length", type=int, default=None, help="How many local frames to perturb; default means until the end")
    parser.add_argument("--eef_jitter_pos_std", type=float, default=0.01, help="Gaussian std for xyz jitter in world units")
    parser.add_argument("--eef_jitter_quat_std", type=float, default=0.01, help="Gaussian std for xyzw jitter before renormalization")
    parser.add_argument("--eef_perturb_seed", type=int, default=1234)
    parser.add_argument("--gripper_perturb_mode", type=str, default="hold", choices=["none", "hold"])
    parser.add_argument("--gripper_perturb_start", type=int, default=0, help="0-based local frame index within the selected range")
    parser.add_argument("--gripper_perturb_length", type=int, default=500, help="How many local frames to override gripper state; default means until the end")
    parser.add_argument("--left_gripper_state", type=str, default="open", choices=["keep", "open", "close"])
    parser.add_argument("--right_gripper_state", type=str, default="open", choices=["keep", "open", "close"])
    parser.add_argument("--gripper_open_value", type=float, default=35.0, help="Numeric gripper value used for the fully open state")
    parser.add_argument("--gripper_close_value", type=float, default=120.0, help="Numeric gripper value used for the fully closed state")
    parser.add_argument(
        "--base_model_paths",
        nargs="+",
        default=[
            "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
            "/mnt/workspace/zsq/Wan_model/Wan2.2-Fun-A14B-Control/models_t5_umt5-xxl-enc-bf16.pth",
            "/mnt/workspace/zsq/Wan_model/Wan2.2-Fun-A14B-Control/Wan2.1_VAE.pth",
        ],
    )
    parser.add_argument("--vace_model_path", type=str, default='/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.1-VACE-1.3B-multiview-perspective-continue10/epoch-79.safetensors')
    args = parser.parse_args()

    if args.original_hz % args.target_hz != 0:
        raise ValueError("--original_hz must be divisible by --target_hz.")
    if args.num_frames != args.predict_frames + 1:
        raise ValueError("--num_frames must equal --predict_frames + 1.")
    if args.start_chunk < 1:
        raise ValueError("--start_chunk must be >= 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    episodes = discover_episodes(args.val_path)
    if args.episode_key not in episodes:
        raise RuntimeError(f"Episode {args.episode_key} not found in {args.val_path}")

    downsample_step = args.original_hz // args.target_hz
    ep_info = build_episode_info(episodes[args.episode_key], downsample_step, args.camera_names)
    total = ep_info["T_ds"]
    if total <= 0:
        raise RuntimeError("No frames available to compose comparison video.")

    total_prediction_frames = max(total - 1, 0)
    total_chunks = (total_prediction_frames + args.predict_frames - 1) // args.predict_frames if total_prediction_frames > 0 else 0
    start_chunk_idx = args.start_chunk - 1
    if start_chunk_idx > total_chunks:
        raise RuntimeError(f"start_chunk={args.start_chunk} exceeds total_chunks={total_chunks}")
    if args.num_chunks is None:
        end_chunk_idx = total_chunks
    else:
        end_chunk_idx = min(start_chunk_idx + args.num_chunks, total_chunks)

    start_frame = 0 if total_chunks == 0 else start_chunk_idx * args.predict_frames
    end_frame = total if total_chunks == 0 else min(1 + end_chunk_idx * args.predict_frames, total)
    print(
        f"[INFO] total_frames={total} total_chunks={total_chunks} "
        f"using_chunks={args.start_chunk}-{end_chunk_idx} "
        f"frame_range=[{start_frame}, {end_frame})"
    )
    if args.gripper_perturb_mode != "none":
        print(
            f"[INFO] gripper_perturb mode={args.gripper_perturb_mode} "
            f"start={args.gripper_perturb_start} length={args.gripper_perturb_length} "
            f"left={args.left_gripper_state} right={args.right_gripper_state} "
            f"open_value={args.gripper_open_value} close_value={args.gripper_close_value}"
        )

    selected_slice = slice(start_frame, end_frame)
    print("[INFO] Loading only selected GT frames...", flush=True)
    gt_frames, raw_hw_by_cam = load_gt_frames_range(
        ep_info, args.camera_names, selected_slice, args.view_height, args.view_width
    )
    print(f"[INFO] Loaded {len(gt_frames)} GT frames for comparison", flush=True)
    print("[INFO] Building action_map/raymap only for selected range...", flush=True)
    action_maps, ray_map_o, ray_map_d = build_controls_range(
        ep_info, args.camera_names, selected_slice, raw_hw_by_cam, args.view_height, args.view_width, args
    )
    print(f"[INFO] Built {len(action_maps)} control frames", flush=True)

    pipe = load_pipe(args.base_model_paths, args.vace_model_path, args.device)
    selected_total = len(gt_frames)
    generated_frames = [gt_frames[0]] + [None] * (selected_total - 1)
    current_context = gt_frames[0]
    total_height = args.view_height * len(args.camera_names)

    for chunk_idx in range(start_chunk_idx, end_chunk_idx):
        context_idx = chunk_idx * args.predict_frames
        local_context_idx = context_idx - start_frame
        valid_prediction_count = min(args.predict_frames, total_prediction_frames - context_idx)
        print(
            f"[INFO] episode={args.episode_key} chunk={chunk_idx + 1}/{total_chunks} "
            f"context_idx={context_idx} valid_prediction_count={valid_prediction_count}"
        )

        control_chunk = get_chunk_with_pad(action_maps, local_context_idx, args.num_frames)
        pipe_kwargs = {
            "prompt": args.prompt,
            "vace_video": control_chunk,
            "vace_reference_image": current_context,
            "height": total_height,
            "width": args.view_width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed + chunk_idx,
            "tiled": True,
        }
        if args.output_raymap:
            pipe_kwargs["ray_map_o"] = get_chunk_with_pad(ray_map_o, local_context_idx, args.num_frames)
            pipe_kwargs["ray_map_d"] = get_chunk_with_pad(ray_map_d, local_context_idx, args.num_frames)

        generated = pipe(**pipe_kwargs)
        new_frames = generated[1: 1 + valid_prediction_count]
        for offset, frame in enumerate(new_frames, start=1):
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(np.array(frame))
            target_idx = local_context_idx + offset
            if target_idx < len(generated_frames):
                generated_frames[target_idx] = frame
        next_context_idx = min(local_context_idx + valid_prediction_count, len(generated_frames) - 1)
        if generated_frames[next_context_idx] is None:
            raise RuntimeError(f"Missing generated context frame at index {next_context_idx}.")
        current_context = generated_frames[next_context_idx]

    episode_out_dir = os.path.join(args.output_dir, args.episode_key)
    os.makedirs(episode_out_dir, exist_ok=True)
    out_path = os.path.join(episode_out_dir, args.output_name)

    writer = imageio.get_writer(out_path, fps=args.fps)
    for i in range(selected_total):
        if generated_frames[i] is None:
            raise RuntimeError(f"Missing generated frame at local index {i}.")
        parts = [
            np.array(gt_frames[i]),
            np.array(generated_frames[i]),
            np.array(action_maps[i]),
        ]
        if args.output_raymap:
            parts.extend([np.array(ray_map_o[i]), np.array(ray_map_d[i])])
        writer.append_data(np.concatenate(parts, axis=1))
    writer.close()
    print(f"[DONE] Saved comparison video to {out_path}")


if __name__ == "__main__":
    main()
