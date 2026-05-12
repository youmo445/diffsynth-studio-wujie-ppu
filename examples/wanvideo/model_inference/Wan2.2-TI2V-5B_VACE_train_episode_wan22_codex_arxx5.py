"""
Wan2.2 Codex VACE inference on one ARXX5 training episode.

This script is aligned with Arxx5Dataset4Wancontrolmultiview training logic:
- traj_map: per-camera actions + per-camera w2c
- ray_map: center-base midpoint frame + per-camera w2c
- raymap_mode: image
- traj_radius_mode: perspective
- camera order: head, left_wrist, right_wrist (height concat)

Example:
    cd /mnt/workspace/zsq/DiffSynth-Studio
    python examples/wanvideo/model_inference/Wan2.2-TI2V-5B_VACE_train_episode_wan22_codex_arxx5.py \
        --agx_base /mnt/data/zsq/agx \
        --episode_pos 0 \
        --output_dir /mnt/workspace/zsq/DiffSynth-Studio/outputs/arxx5_train_episode_infer
"""

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from diffsynth.data.video import save_video
from diffsynth.trainers.arxx5_dataset4_wancontrolmultiview import (
    CAMERAS,
    generate_raymap,
    generate_traj_map,
    get_color_intrinsics,
    load_calib_extrinsics,
    make_center_ray_w2c_sequences,
    make_per_camera_sequences,
    read_video_rgb_by_indices,
    scale_intrinsic,
    to_uint8_img,
)

try:
    import pyarrow.parquet as pq
except Exception as exc:
    raise ImportError("This script needs pyarrow: pip install pyarrow") from exc


def discover_episodes(agx_base, camera_names):
    base = Path(agx_base)
    data_dir = base / "data"
    videos_dir = base / "videos"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    episodes = []
    for parquet_path in sorted(data_dir.glob("chunk-*/episode_*.parquet")):
        m = re.search(r"episode_(\d+)\.parquet$", parquet_path.name)
        if m is None:
            continue
        ep_idx = int(m.group(1))
        chunk = parquet_path.parent.name
        video_paths = {
            cam: videos_dir / chunk / f"observation.images.{cam}" / f"episode_{ep_idx:06d}.mp4"
            for cam in CAMERAS
        }
        if all(video_paths[cam].exists() for cam in camera_names):
            episodes.append(
                {
                    "episode_index": ep_idx,
                    "chunk": chunk,
                    "parquet_path": parquet_path,
                    "video_paths": video_paths,
                }
            )
    return episodes


def load_episode_arrays(parquet_path):
    table = pq.read_table(parquet_path, columns=["observation.ee_pose", "observation.state"])
    ee_pose = np.asarray(table["observation.ee_pose"].to_pylist(), dtype=np.float32)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    return ee_pose, state


def get_video_len(path):
    import av

    container = av.open(str(path))
    try:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            return 0
        if stream.frames is not None and int(stream.frames) > 0:
            return int(stream.frames)
        count = 0
        for _ in container.decode(stream):
            count += 1
        return count
    finally:
        container.close()


def select_episode(episodes, episode_index=None, episode_pos=0):
    if episode_index is not None:
        cand = [ep for ep in episodes if ep["episode_index"] == episode_index]
        if not cand:
            raise RuntimeError(f"episode_index={episode_index} not found")
        return cand[0]
    if not episodes:
        raise RuntimeError("No episodes found.")
    if episode_pos < 0 or episode_pos >= len(episodes):
        raise RuntimeError(f"episode_pos out of range: {episode_pos}, total={len(episodes)}")
    return episodes[episode_pos]


def build_episode_info(ep, args):
    ee_pose, state = load_episode_arrays(ep["parquet_path"])
    n = min(ee_pose.shape[0], state.shape[0])
    for cam in args.camera_names:
        n = min(n, get_video_len(ep["video_paths"][cam]))

    if n <= 0:
        raise RuntimeError("Empty episode.")

    ds_idx = np.arange(0, n, args.downsample_step, dtype=np.int64)
    ee_ds = ee_pose[ds_idx]
    state_ds = state[ds_idx]

    extr = load_calib_extrinsics(
        Path(args.head_left_calib),
        Path(args.head_right_calib),
        Path(args.left_calib),
        Path(args.right_calib),
    )
    traj_actions_by_cam, traj_w2c_by_cam = make_per_camera_sequences(ee_ds, state_ds, extr, args.camera_axis_mode)
    ray_w2c_by_cam = make_center_ray_w2c_sequences(ee_ds, extr, args.camera_axis_mode)

    return {
        "episode_index": ep["episode_index"],
        "chunk": ep["chunk"],
        "parquet_path": ep["parquet_path"],
        "video_paths": ep["video_paths"],
        "ds_indices": ds_idx,
        "T_ds": len(ds_idx),
        "traj_actions_by_cam": traj_actions_by_cam,
        "traj_w2c_by_cam": traj_w2c_by_cam,
        "ray_w2c_by_cam": ray_w2c_by_cam,
    }


def load_first_multiview_frame(ep_info, camera_names, out_h, out_w):
    frames = []
    first_raw_idx = int(ep_info["ds_indices"][0])
    for cam in camera_names:
        frame = read_video_rgb_by_indices(ep_info["video_paths"][cam], [first_raw_idx])[0]
        frame = np.array(Image.fromarray(frame).resize((out_w, out_h), Image.BILINEAR))
        frames.append(frame)
    return Image.fromarray(np.concatenate(frames, axis=0))


def build_episode_controls(ep_info, args):
    camera_names = args.camera_names
    out_h = args.view_height
    out_w = args.view_width

    intrinsics = get_color_intrinsics()
    total_frames = ep_info["T_ds"]

    control_per_cam = {}
    ray_o_per_cam = {}
    ray_d_per_cam = {}

    first_raw_idx = int(ep_info["ds_indices"][0])

    for cam in camera_names:
        sample_frame = read_video_rgb_by_indices(ep_info["video_paths"][cam], [first_raw_idx])[0]
        in_h, in_w = sample_frame.shape[:2]

        k = scale_intrinsic(intrinsics[cam], in_h, in_w, out_h, out_w)

        traj_actions = ep_info["traj_actions_by_cam"][cam]
        traj_w2c = ep_info["traj_w2c_by_cam"][cam]
        control_per_cam[cam] = generate_traj_map(
            traj_actions,
            traj_w2c,
            k,
            out_h,
            out_w,
            radius=args.traj_radius,
            radius_mode=args.traj_radius_mode,
            min_radius=args.traj_min_radius,
            max_radius=args.traj_max_radius,
            ref_depth=args.traj_ref_depth,
            near_depth=args.traj_near_depth,
            far_depth=args.traj_far_depth,
        )

        ray_o_seq = []
        ray_d_seq = []
        ray_w2c = ep_info["ray_w2c_by_cam"][cam]
        for t in range(total_frames):
            c2w = np.linalg.inv(ray_w2c[t]).astype(np.float32)
            ro, rd = generate_raymap(k, c2w, out_h, out_w)
            ray_o_seq.append(to_uint8_img(ro, args.ray_o_vmin, args.ray_o_vmax))
            ray_d_seq.append(to_uint8_img(rd, args.ray_d_vmin, args.ray_d_vmax))
        ray_o_per_cam[cam] = ray_o_seq
        ray_d_per_cam[cam] = ray_d_seq

    control_video = []
    ray_map_o = []
    ray_map_d = []

    for t in range(total_frames):
        control_concat = np.concatenate([control_per_cam[cam][t] for cam in camera_names], axis=0)
        ray_o_concat = np.concatenate([ray_o_per_cam[cam][t] for cam in camera_names], axis=0)
        ray_d_concat = np.concatenate([ray_d_per_cam[cam][t] for cam in camera_names], axis=0)

        control_video.append(Image.fromarray(control_concat))
        ray_map_o.append(Image.fromarray(ray_o_concat))
        ray_map_d.append(Image.fromarray(ray_d_concat))

    return control_video, ray_map_o, ray_map_d


def get_chunk_with_pad(frames, start_idx, num_frames):
    chunk = frames[start_idx: start_idx + num_frames]
    if len(chunk) < num_frames:
        chunk = chunk + [frames[-1]] * (num_frames - len(chunk))
    return chunk


def load_pipe(base_model_paths, vace_model_path, device, vace_in_dim=144):
    import torch

    from diffsynth.models.wan_video_vace_wan22_codex import VaceWan22CodexModel
    from diffsynth.pipelines.wan_video_new_wan22_codex import ModelConfig, WanVideoPipeline
    from diffsynth.models.utils import load_state_dict

    model_configs = [ModelConfig(path=path) for path in base_model_paths]
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
    )
    if vace_model_path:
        if getattr(pipe, "vace", None) is None:
            pipe.vace = VaceWan22CodexModel(vace_in_dim=vace_in_dim).to(dtype=pipe.torch_dtype, device=pipe.device)
        state_dict = load_state_dict(vace_model_path)
        if any(name.startswith("vace_global_") for name in state_dict):
            pipe.vace.enable_global_cross_attn(global_context_dim=16)
        pipe.vace.load_state_dict(state_dict)
    return pipe


def save_frames(frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame.save(os.path.join(out_dir, f"frame_{idx:06d}.png"))


def run_episode(pipe, ep_info, args):
    import torch

    total_height = args.view_height * len(args.camera_names)
    first_frame = load_first_multiview_frame(ep_info, args.camera_names, args.view_height, args.view_width)
    control_video, ray_map_o, ray_map_d = build_episode_controls(ep_info, args)

    total_frames = ep_info["T_ds"]
    if total_frames <= 0:
        raise RuntimeError("Episode has no downsampled frames.")

    ep_tag = f"episode_{ep_info['episode_index']:06d}"
    ep_out_dir = os.path.join(args.output_dir, ep_tag)
    images_dir = os.path.join(ep_out_dir, "images")
    os.makedirs(ep_out_dir, exist_ok=True)

    if args.save_control_video:
        save_video(control_video, os.path.join(ep_out_dir, "control_video.mp4"), fps=args.fps, quality=5)
    if args.save_raymap_video:
        save_video(ray_map_o, os.path.join(ep_out_dir, "ray_map_o.mp4"), fps=args.fps, quality=5)
        save_video(ray_map_d, os.path.join(ep_out_dir, "ray_map_d.mp4"), fps=args.fps, quality=5)

    generated_frames = [first_frame]
    current_context = first_frame

    total_prediction_frames = max(total_frames - 1, 0)
    num_chunks = math.ceil(total_prediction_frames / args.predict_frames) if total_prediction_frames > 0 else 0
    total_gen_time = 0.0
    total_gen_frames = 0

    for chunk_idx in range(num_chunks):
        context_idx = chunk_idx * args.predict_frames
        valid_prediction_count = min(args.predict_frames, total_prediction_frames - context_idx)

        print(
            f"[INFO] {ep_tag} chunk={chunk_idx + 1}/{num_chunks} "
            f"context_idx={context_idx} valid_prediction_count={valid_prediction_count}"
        )

        pipe_kwargs = {
            "prompt": args.prompt,
            "input_image": current_context,
            "vace_video": get_chunk_with_pad(control_video, context_idx, args.num_frames),
            "ray_map_o": get_chunk_with_pad(ray_map_o, context_idx, args.num_frames),
            "ray_map_d": get_chunk_with_pad(ray_map_d, context_idx, args.num_frames),
            "height": total_height,
            "width": args.view_width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed + chunk_idx,
            "tiled": args.tiled,
        }

        if args.dry_run:
            print(
                f"[DRY RUN] {ep_tag} chunk={chunk_idx + 1}/{num_chunks} "
                f"context_idx={context_idx} valid_prediction_count={valid_prediction_count}"
            )
            break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        generated = pipe(**pipe_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        chunk_time = time.perf_counter() - t0

        new_frames = generated[1: 1 + valid_prediction_count]
        for frame in new_frames:
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(np.array(frame))
            generated_frames.append(frame)
        current_context = generated_frames[-1]

        chunk_fps = valid_prediction_count / max(chunk_time, 1e-8)
        total_gen_time += chunk_time
        total_gen_frames += valid_prediction_count
        avg_fps = total_gen_frames / max(total_gen_time, 1e-8)
        print(
            f"[SPEED] {ep_tag} chunk={chunk_idx + 1}/{num_chunks} "
            f"time={chunk_time:.3f}s frames={valid_prediction_count} "
            f"chunk_fps={chunk_fps:.3f} avg_fps={avg_fps:.3f}"
        )

    if not args.dry_run and len(generated_frames) != total_frames:
        raise RuntimeError(f"Generated {len(generated_frames)} frames, expected {total_frames}.")

    save_frames(generated_frames, images_dir)
    save_video(generated_frames, os.path.join(ep_out_dir, "video.mp4"), fps=args.fps, quality=5)
    print(f"[DONE] {ep_tag} saved_frames={len(generated_frames)} output_dir={ep_out_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--agx_base", type=str, default="/mnt/data/zsq/agx")
    parser.add_argument("--output_dir", type=str, default="/mnt/workspace/zsq/DiffSynth-Studio/outputs/arxx5_train_episode_wan22_codex")

    parser.add_argument("--prompt", type=str, default="机械臂按照要求移动夹爪执行任务")
    parser.add_argument("--camera_names", nargs="+", default=["head", "left_wrist", "right_wrist"])
    parser.add_argument("--camera_axis_mode", type=str, default="identity", choices=["identity", "fixed", "frames3d"])

    parser.add_argument("--original_hz", type=int, default=30)
    parser.add_argument("--target_hz", type=int, default=30)
    parser.add_argument("--num_frames", type=int, default=9)
    parser.add_argument("--predict_frames", type=int, default=8)

    parser.add_argument("--view_height", type=int, default=160)
    parser.add_argument("--view_width", type=int, default=224)

    parser.add_argument("--traj_radius", type=int, default=40)
    parser.add_argument("--traj_radius_mode", type=str, default="perspective", choices=["constant", "perspective", "depth_norm"])
    parser.add_argument("--traj_min_radius", type=int, default=14)
    parser.add_argument("--traj_max_radius", type=int, default=58)
    parser.add_argument("--traj_ref_depth", type=float, default=0.30)
    parser.add_argument("--traj_near_depth", type=float, default=0.19)
    parser.add_argument("--traj_far_depth", type=float, default=0.69)

    parser.add_argument("--ray_o_vmin", type=float, default=-1.5)
    parser.add_argument("--ray_o_vmax", type=float, default=1.5)
    parser.add_argument("--ray_d_vmin", type=float, default=-1.0)
    parser.add_argument("--ray_d_vmax", type=float, default=1.0)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vace_in_dim", type=int, default=144)

    parser.add_argument("--episode_index", type=int, default=None)
    parser.add_argument("--episode_pos", type=int, default=0)

    parser.add_argument("--head_left_calib", type=str, default="/mnt/data/zsq/outputs/calib_eye_to_hand_head_left/result_eye_to_hand.json")
    parser.add_argument("--head_right_calib", type=str, default="/mnt/data/zsq/outputs/calib_eye_to_hand_head_right/result_eye_to_hand.json")
    parser.add_argument("--left_calib", type=str, default="/mnt/data/zsq/outputs/calib_eye_in_hand_left/result_eye_in_hand.json")
    parser.add_argument("--right_calib", type=str, default="/mnt/data/zsq/outputs/calib_eye_in_hand_right/result_eye_in_hand.json")

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--save_control_video", action="store_true")
    parser.add_argument("--save_raymap_video", action="store_true")

    parser.add_argument(
        "--base_model_paths",
        type=str,
        default=json.dumps(
            [
                [
                    "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
                    "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
                    "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors",
                ],
                "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
                "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
            ]
        ),
        help="JSON list for base model paths: [ [dit_shard1,dit_shard2,dit_shard3], t5_path, vae_path ]",
    )
    parser.add_argument(
        "--vace_model_path",
        type=str,
        default="/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.2-TI2V-5B-VACE-wan22-codex-ray-action-perspective-arxx5/epoch-19.safetensors",
        help="Path to VACE-only checkpoint exported with remove_prefix_in_ckpt=pipe.vace.",
    )

    args = parser.parse_args()

    if args.num_frames != args.predict_frames + 1:
        raise ValueError("--num_frames must equal --predict_frames + 1.")
    if args.original_hz % args.target_hz != 0:
        raise ValueError("--original_hz must be divisible by --target_hz.")
    args.base_model_paths = json.loads(args.base_model_paths)

    args.downsample_step = args.original_hz // args.target_hz

    episodes = discover_episodes(args.agx_base, args.camera_names)
    print(f"[INFO] discovered episodes: {len(episodes)}")
    selected = select_episode(episodes, args.episode_index, args.episode_pos)
    print(
        f"[INFO] selected episode_index={selected['episode_index']} "
        f"chunk={selected['chunk']} parquet={selected['parquet_path']}"
    )

    ep_info = build_episode_info(selected, args)

    raymap_tag = "raymap"
    out_tag = f"{args.traj_radius_mode}_{raymap_tag}_hz{args.original_hz}to{args.target_hz}"
    args.output_dir = f"{args.output_dir}_{out_tag}"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] output_dir={args.output_dir}")

    if args.dry_run:
        pipe = None
    else:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this script.")
        pipe = load_pipe(args.base_model_paths, args.vace_model_path, args.device, args.vace_in_dim)

    run_episode(pipe, ep_info, args)


if __name__ == "__main__":
    main()
