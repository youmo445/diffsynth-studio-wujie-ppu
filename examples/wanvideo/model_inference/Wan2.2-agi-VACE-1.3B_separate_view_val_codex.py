"""
Run validation inference with separate single-view VACE checkpoints, then stitch
the generated views back into the same vertical multiview layout produced by
Wan2.2-agi-VACE-1.3B_multiview_codex.py.
"""

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
MULTIVIEW_SCRIPT = SCRIPT_DIR / "Wan2.2-agi-VACE-1.3B_multiview_codex.py"


def load_multiview_module():
    spec = importlib.util.spec_from_file_location("wan_multiview_codex_impl", str(MULTIVIEW_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discover_episodes(base_path):
    proprio_base = os.path.join(base_path, "proprio_stats")
    episodes = []
    if not os.path.isdir(proprio_base):
        raise RuntimeError(f"Only split Agi2024 layout is supported here: {base_path}")
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


def parse_devices(devices):
    if devices.strip().lower() == "auto":
        return [str(i) for i in range(16)]
    return [d.strip() for d in devices.split(",") if d.strip()]


def split_tasks(tasks, num_workers):
    buckets = [[] for _ in range(num_workers)]
    for idx, task in enumerate(tasks):
        buckets[idx % num_workers].append(task)
    return [bucket for bucket in buckets if bucket]


def save_frames(frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(np.asarray(frame))
        frame.save(os.path.join(out_dir, f"frame_{idx:06d}.png"))


def save_video_stream(frame_paths, out_path, fps):
    writer = imageio.get_writer(out_path, fps=fps)
    try:
        for p in frame_paths:
            writer.append_data(imageio.imread(str(p)))
    finally:
        writer.close()


def run_single_view_episode(mv, pipe, ep, cam, args_dict, view_out_dir):
    os.makedirs(view_out_dir, exist_ok=True)
    images_dir = os.path.join(view_out_dir, "images")
    done_path = os.path.join(view_out_dir, "_done.json")
    if args_dict["skip_existing"] and os.path.exists(done_path):
        return

    downsample_step = args_dict["original_hz"] // args_dict["target_hz"]
    ep_info = mv.build_episode_info(ep, downsample_step, [cam])
    view_height = args_dict["view_height"]
    view_width = args_dict["view_width"]
    total_frames = ep_info["T_ds"]
    if total_frames <= 0:
        raise RuntimeError(f"Episode {ep['episode_key']} has no frames.")

    first_frame = mv.load_first_multiview_frame(ep_info, [cam], view_height, view_width)
    control_video, ray_map_o, ray_map_d = mv.build_episode_controls(
        ep_info,
        [cam],
        view_height,
        view_width,
        argparse.Namespace(**args_dict),
    )

    generated_frames = [first_frame]
    current_context = first_frame
    predict_frames = args_dict["predict_frames"]
    fixed_num_frames = args_dict["num_frames"]
    total_prediction_frames = max(total_frames - 1, 0)
    num_chunks = math.ceil(total_prediction_frames / predict_frames) if total_prediction_frames > 0 else 0

    for chunk_idx in range(num_chunks):
        context_idx = chunk_idx * predict_frames
        valid_prediction_count = min(predict_frames, total_prediction_frames - context_idx)
        control_chunk = mv.get_chunk_with_pad(control_video, context_idx, fixed_num_frames)
        pipe_kwargs = {
            "prompt": args_dict["prompt"],
            "vace_video": control_chunk,
            "vace_reference_image": current_context,
            "height": view_height,
            "width": view_width,
            "num_frames": fixed_num_frames,
            "num_inference_steps": args_dict["num_inference_steps"],
            "cfg_scale": args_dict["cfg_scale"],
            "seed": args_dict["seed"] + chunk_idx,
            "tiled": True,
        }
        if args_dict["output_raymap"]:
            pipe_kwargs["ray_map_o"] = mv.get_chunk_with_pad(ray_map_o, context_idx, fixed_num_frames)
            pipe_kwargs["ray_map_d"] = mv.get_chunk_with_pad(ray_map_d, context_idx, fixed_num_frames)

        generated = pipe(**pipe_kwargs)
        new_frames = generated[1: 1 + valid_prediction_count]
        for frame in new_frames:
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(np.asarray(frame))
            generated_frames.append(frame)
        current_context = generated_frames[-1]

    if len(generated_frames) != total_frames:
        raise RuntimeError(
            f"{ep['episode_key']} {cam}: generated {len(generated_frames)} frames, expected {total_frames}."
        )

    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    save_frames(generated_frames, images_dir)
    with open(done_path, "w", encoding="utf-8") as f:
        json.dump({"episode_key": ep["episode_key"], "camera": cam, "frames": len(generated_frames)}, f)


def worker_main(worker_id, device, tasks, args_dict, model_paths):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    mv = load_multiview_module()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"Worker {worker_id}: CUDA is not available on visible device {device}.")

    print(f"[WORKER {worker_id}] device={device} tasks={len(tasks)}", flush=True)
    for cam in args_dict["camera_names"]:
        cam_tasks = [task for task in tasks if task["camera"] == cam]
        if not cam_tasks:
            continue

        print(f"[WORKER {worker_id}] loading camera={cam} model={model_paths[cam]}", flush=True)
        pipe = mv.load_pipe(args_dict["base_model_paths"], model_paths[cam], "cuda:0")
        for task in cam_tasks:
            ep = task["episode"]
            view_out_dir = os.path.join(args_dict["view_cache_dir"], ep["episode_key"], cam)
            print(f"[WORKER {worker_id}] episode={ep['episode_key']} camera={cam}", flush=True)
            run_single_view_episode(mv, pipe, ep, cam, args_dict, view_out_dir)
        del pipe
        torch.cuda.empty_cache()


def stitch_episode(episode_key, camera_names, view_cache_dir, final_out_dir, fps, skip_existing):
    ep_out_dir = os.path.join(final_out_dir, episode_key)
    images_dir = os.path.join(ep_out_dir, "images")
    done_path = os.path.join(ep_out_dir, "_stitched_done.json")
    if skip_existing and os.path.exists(done_path):
        return

    cam_frame_paths = []
    for cam in camera_names:
        cam_images = sorted((Path(view_cache_dir) / episode_key / cam / "images").glob("frame_*.png"))
        if not cam_images:
            raise RuntimeError(f"Missing generated frames for {episode_key} {cam}")
        cam_frame_paths.append(cam_images)

    total_frames = min(len(paths) for paths in cam_frame_paths)
    if total_frames <= 0:
        raise RuntimeError(f"No overlapping frames for {episode_key}")

    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir, exist_ok=True)

    stitched_paths = []
    for idx in range(total_frames):
        parts = [imageio.imread(str(paths[idx])) for paths in cam_frame_paths]
        stitched = np.concatenate(parts, axis=0)
        out_path = Path(images_dir) / f"frame_{idx:06d}.png"
        Image.fromarray(stitched).save(out_path)
        stitched_paths.append(out_path)

    save_video_stream(stitched_paths, os.path.join(ep_out_dir, "video.mp4"), fps=fps)
    with open(done_path, "w", encoding="utf-8") as f:
        json.dump({"episode_key": episode_key, "frames": total_frames, "cameras": camera_names}, f)
    print(f"[STITCH] {episode_key}: frames={total_frames} -> {ep_out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", type=str, default="/mnt/workspace/zsq/Agi2024subset_split/val")
    parser.add_argument("--output_dir", type=str, default="/mnt/workspace/zsq/Agi2024subset_split/val_gen_separate_views")
    parser.add_argument("--prompt", type=str, default="机械臂按照要求移动夹爪执行任务")
    parser.add_argument("--camera_names", nargs="+", default=["head", "hand_left", "hand_right"])
    parser.add_argument("--model_path_mode", type=str, default="separate", choices=["separate", "shared"])
    parser.add_argument("--shared_model_path", type=str, default="/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.1-VACE-1.3B-random-view-raymap-perspectivate/epoch-59.safetensors")
    parser.add_argument("--head_model_path", type=str, default="/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.1-VACE-1.3B-head-raymap-perspectivate/epoch-59.safetensors")
    parser.add_argument("--hand_left_model_path", type=str, default="/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.1-VACE-1.3B-hand_left-raymap-perspectivate/epoch-59.safetensors")
    parser.add_argument("--hand_right_model_path", type=str, default="/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.1-VACE-1.3B-hand_right-raymap-perspectivate/epoch-59.safetensors")
    parser.add_argument("--devices", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--original_hz", type=int, default=30)
    parser.add_argument("--target_hz", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=9)
    parser.add_argument("--predict_frames", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
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
    parser.add_argument("--output_raymap", action="store_true")
    parser.add_argument("--ray_o_vmin", type=float, default=-1.5)
    parser.add_argument("--ray_o_vmax", type=float, default=1.5)
    parser.add_argument("--ray_d_vmin", type=float, default=-1.0)
    parser.add_argument("--ray_d_vmax", type=float, default=1.0)
    parser.add_argument("--episode_limit", type=int, default=None)
    parser.add_argument("--episode_filter", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--base_model_paths",
        nargs="+",
        default=[
            "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
            "/mnt/workspace/zsq/Wan_model/Wan2.2-Fun-A14B-Control/models_t5_umt5-xxl-enc-bf16.pth",
            "/mnt/workspace/zsq/Wan_model/Wan2.2-Fun-A14B-Control/Wan2.1_VAE.pth",
        ],
    )
    args = parser.parse_args()

    if args.num_frames != args.predict_frames + 1:
        raise ValueError("--num_frames must equal --predict_frames + 1.")
    if args.original_hz % args.target_hz != 0:
        raise ValueError("--original_hz must be divisible by --target_hz.")

    if args.model_path_mode == "shared":
        model_paths = {cam: args.shared_model_path for cam in args.camera_names}
    else:
        model_paths = {
            "head": args.head_model_path,
            "hand_left": args.hand_left_model_path,
            "hand_right": args.hand_right_model_path,
        }
    for cam in args.camera_names:
        if cam not in model_paths:
            raise ValueError(f"No model path argument is defined for camera: {cam}")
        if not os.path.exists(model_paths[cam]):
            raise FileNotFoundError(f"Missing model for {cam}: {model_paths[cam]}")

    raymap_tag = "raymap" if args.output_raymap else "noraymap"
    final_out_dir = f"{args.output_dir}_{args.traj_radius_mode}_{raymap_tag}"
    view_cache_dir = os.path.join(final_out_dir, ".view_cache")
    os.makedirs(view_cache_dir, exist_ok=True)
    os.makedirs(final_out_dir, exist_ok=True)

    episodes = discover_episodes(args.val_path)
    if args.episode_filter is not None:
        episodes = [ep for ep in episodes if ep["episode_key"] == args.episode_filter]
    if args.episode_limit is not None:
        episodes = episodes[: args.episode_limit]
    if not episodes:
        raise RuntimeError("No episodes selected.")

    tasks = [{"episode": ep, "camera": cam} for ep in episodes for cam in args.camera_names]
    devices = parse_devices(args.devices)
    num_workers = min(args.max_workers, len(devices), len(tasks))
    task_buckets = split_tasks(tasks, num_workers)

    args_dict = vars(args).copy()
    args_dict["view_cache_dir"] = view_cache_dir
    args_dict["camera_names"] = list(args.camera_names)
    print(
        f"[INFO] episodes={len(episodes)} cameras={args.camera_names} tasks={len(tasks)} "
        f"workers={len(task_buckets)} output={final_out_dir}",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    procs = []
    for worker_id, bucket in enumerate(task_buckets):
        device = devices[worker_id % len(devices)]
        p = ctx.Process(target=worker_main, args=(worker_id, device, bucket, args_dict, model_paths))
        p.start()
        procs.append(p)

    failed = []
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append(p.exitcode)
    if failed:
        raise RuntimeError(f"{len(failed)} worker(s) failed: {failed}")

    for ep in episodes:
        stitch_episode(
            ep["episode_key"],
            list(args.camera_names),
            view_cache_dir,
            final_out_dir,
            args.fps,
            args.skip_existing,
        )
    print(f"[DONE] Results saved to {final_out_dir}", flush=True)


if __name__ == "__main__":
    main()
