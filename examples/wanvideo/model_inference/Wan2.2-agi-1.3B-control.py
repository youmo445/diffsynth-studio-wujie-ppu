"""
AgiBotWC validation-set autoregressive inference for Wan2.2-A14B-Control.

Features:
1. Keep every inference chunk at a fixed length: 1 context frame + predict_frames.
2. Left-pad the trajectory sequence by one frame so chunk boundaries stay uniform.
3. Right-pad the last chunk if needed, then drop padded generated tail frames.
4. Distribute episodes across multiple GPUs, one episode per GPU at a time.

Example:
    python inference_agibot_val_multi_gpu.py \
        --val_path /mnt/workspace/zsq/AgibotWCsubset/val \
        --output_dir /mnt/workspace/zsq/AgibotWCsubset/val_results \
        --original_hz 30 \
        --target_hz 5 \
        --predict_frames 8 \
        --gpu_ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
        --seed 42
"""

import argparse
import math
import os
import json
import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image, ImageDraw
from collections import defaultdict
from scipy.spatial.transform import Rotation
from decord import VideoReader, cpu
import matplotlib.cm as cm


# ============================================================
# Trajectory map generation
# ============================================================

_EVAC_ColorMapLeft = cm.Greens
_EVAC_ColorMapRight = cm.Reds
_EVAC_ColorListLeft = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]
_EVAC_ColorListRight = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]
_EVAC_EndEffectorPts = np.array(
    [
        [0, 0, 0, 1],
        [0.1, 0, 0, 1],
        [0, 0.1, 0, 1],
        [0, 0, 0.1, 1],
    ],
    dtype=np.float32,
)
_EVAC_Gripper2EEFCvt = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.23],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)
_EVAC_EEF2CamLeft = [0, 0, -0.5236]
_EVAC_EEF2CamRight = [0, 0, 0.5236]


def _quaternion_to_matrix_np(wxyz):
    w, x, y, z = wxyz[..., 0], wxyz[..., 1], wxyz[..., 2], wxyz[..., 3]
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    mat = np.stack(
        [
            1.0 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1.0 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1.0 - (txx + tyy),
        ],
        axis=-1,
    ).reshape(wxyz.shape[:-1] + (3, 3))
    return mat


def _get_transformation_matrix_np(xyz_quat):
    rot_quat = xyz_quat[:, 3:]
    wxyz = rot_quat[:, [3, 0, 1, 2]]
    rot = _quaternion_to_matrix_np(wxyz)
    trans = xyz_quat[:, :3]
    num_frames = xyz_quat.shape[0]
    mat = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
    mat[:, :3, :3] = rot
    mat[:, :3, 3] = trans
    return mat


def generate_traj_map(abs_actions, w2c, intrinsic, h, w, radius=50):
    num_frames = abs_actions.shape[0]
    ee_key_pts = _EVAC_EndEffectorPts.T
    cvt = _EVAC_Gripper2EEFCvt
    w2c_mat = w2c.astype(np.float32)

    cvt_vis_l = Rotation.from_euler("xyz", _EVAC_EEF2CamLeft)
    cvt_vis_r = Rotation.from_euler("xyz", _EVAC_EEF2CamRight)

    traj_maps = []
    for t in range(num_frames):
        pos_l = abs_actions[t, 0:3]
        quat_l = abs_actions[t, 3:7]
        rot_l = Rotation.from_quat(quat_l)
        rot_vis_l = rot_l * cvt_vis_l
        vis_quat_l = np.concatenate([pos_l, rot_vis_l.as_quat()])

        pos_r = abs_actions[t, 8:11]
        quat_r = abs_actions[t, 11:15]
        rot_r = Rotation.from_quat(quat_r)
        rot_vis_r = rot_r * cvt_vis_r
        vis_quat_r = np.concatenate([pos_r, rot_vis_r.as_quat()])

        pose_l_mat = _get_transformation_matrix_np(vis_quat_l[None])[0]
        pose_r_mat = _get_transformation_matrix_np(vis_quat_r[None])[0]

        ee2cam_l = w2c_mat @ pose_l_mat @ cvt
        ee2cam_r = w2c_mat @ pose_r_mat @ cvt

        pts_l = (ee2cam_l @ ee_key_pts)[:3, :]
        pts_r = (ee2cam_r @ ee_key_pts)[:3, :]

        uvs_l = (intrinsic @ pts_l)
        uvs_l = (uvs_l[:2, :] / uvs_l[2:3, :]).T.astype(np.int64)
        uvs_r = (intrinsic @ pts_r)
        uvs_r = (uvs_r[:2, :] / uvs_r[2:3, :]).T.astype(np.int64)

        pil_img = Image.new("RGB", (w, h), (50, 50, 50))
        draw = ImageDraw.Draw(pil_img)

        grip_l = abs_actions[t, 7]
        grip_r = abs_actions[t, 15]
        norm_l = np.clip(grip_l / 120.0, 0.0, 1.0)
        norm_r = np.clip(grip_r / 120.0, 0.0, 1.0)
        color_l = tuple(int(c * 255) for c in _EVAC_ColorMapLeft(norm_l)[:3])
        color_r = tuple(int(c * 255) for c in _EVAC_ColorMapRight(norm_r)[:3])

        for uvs, color, color_list in [
            (uvs_l, color_l, _EVAC_ColorListLeft),
            (uvs_r, color_r, _EVAC_ColorListRight),
        ]:
            base = uvs[0]
            bx, by = int(base[0]), int(base[1])
            if 0 <= bx < w and 0 <= by < h:
                draw.ellipse(
                    [bx - radius, by - radius, bx + radius, by + radius],
                    fill=color,
                )
                for k in range(1, len(uvs)):
                    pt = uvs[k]
                    draw.line(
                        [(bx, by), (int(pt[0]), int(pt[1]))],
                        fill=color_list[k - 1],
                        width=8,
                    )

        traj_maps.append(pil_img)

    return traj_maps


# ============================================================
# Episode data loading
# ============================================================


def discover_episodes(val_path):
    episode_dict = defaultdict(list)
    for name in sorted(os.listdir(val_path)):
        path = os.path.join(val_path, name)
        if not os.path.isdir(path):
            continue
        key = "-".join(name.split("-")[:2])
        episode_dict[key].append(path)

    episodes = []
    for key in sorted(episode_dict.keys()):
        episodes.append((key, sorted(episode_dict[key])))
    return episodes


def load_episode_data(ep_paths, downsample_step, height, width):
    pos_list, quat_list, grip_list = [], [], []

    for sub in ep_paths:
        h5_path = os.path.join(sub, "proprio_stats.h5")
        with h5py.File(h5_path, "r") as f:
            pos = f["state/end/position"][:]
            quat = f["state/end/orientation"][:]
            grip = f["state/effector/position"][:]

        num_frames = pos.shape[0]
        pos_list.append(pos)
        quat_list.append(quat)
        grip_list.append(grip.reshape(num_frames, 2, -1)[..., 0] if grip.ndim == 3 else grip)

    pos_all = np.concatenate(pos_list, axis=0)
    quat_all = np.concatenate(quat_list, axis=0)
    grip_all = np.concatenate(grip_list, axis=0)

    total_raw_frames = pos_all.shape[0]
    ds_idx = np.arange(0, total_raw_frames, downsample_step)
    total_downsampled_frames = len(ds_idx)

    pos_ds = pos_all[ds_idx]
    quat_ds = quat_all[ds_idx]
    grip_ds = grip_all[ds_idx]

    abs_actions = np.zeros((total_downsampled_frames, 16), dtype=np.float32)
    for t in range(total_downsampled_frames):
        abs_actions[t, 0:3] = pos_ds[t, 0]
        abs_actions[t, 3:7] = quat_ds[t, 0]
        abs_actions[t, 7] = grip_ds[t, 0]
        abs_actions[t, 8:11] = pos_ds[t, 1]
        abs_actions[t, 11:15] = quat_ds[t, 1]
        abs_actions[t, 15] = grip_ds[t, 1]

    intr_path = os.path.join(ep_paths[0], "head_intrinsic_params.json")
    with open(intr_path, "r", encoding="utf-8") as f:
        info = json.load(f)["intrinsic"]
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = info["fx"]
    intrinsic[1, 1] = info["fy"]
    intrinsic[0, 2] = info["ppx"]
    intrinsic[1, 2] = info["ppy"]

    extr_path = os.path.join(ep_paths[0], "head_extrinsic_params_aligned.json")
    with open(extr_path, "r", encoding="utf-8") as f:
        extr_info = json.load(f)[0]
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.array(extr_info["extrinsic"]["rotation_matrix"], dtype=np.float32)
    c2w[:3, 3] = np.array(extr_info["extrinsic"]["translation_vector"], dtype=np.float32)
    w2c = np.linalg.inv(c2w).astype(np.float32)

    vr = VideoReader(os.path.join(ep_paths[0], "head_color.mp4"), ctx=cpu(0))
    first_frame_np = vr[0].asnumpy()
    del vr
    ori_h, ori_w = first_frame_np.shape[:2]
    first_frame = Image.fromarray(first_frame_np).resize((width, height), Image.BILINEAR)

    h_scale = height / ori_h
    w_scale = width / ori_w
    intrinsic_scaled = intrinsic.copy()
    intrinsic_scaled[0, 0] *= w_scale
    intrinsic_scaled[0, 2] *= w_scale
    intrinsic_scaled[1, 1] *= h_scale
    intrinsic_scaled[1, 2] *= h_scale

    return first_frame, abs_actions, intrinsic_scaled, w2c, total_downsampled_frames


# ============================================================
# Inference helpers
# ============================================================


def parse_gpu_ids(gpu_ids_arg):
    if gpu_ids_arg is None or gpu_ids_arg.strip() == "":
        return list(range(torch.cuda.device_count()))
    return [int(x.strip()) for x in gpu_ids_arg.split(",") if x.strip()]


def build_episode_assignments(episodes, gpu_ids):
    assignments = [[] for _ in gpu_ids]
    for idx, episode in enumerate(episodes):
        assignments[idx % len(gpu_ids)].append(episode)
    return assignments


def build_fixed_chunk_control_video(all_traj_maps, chunk_start, num_frames):
    padded_traj_maps = [all_traj_maps[0]] + all_traj_maps
    chunk = padded_traj_maps[chunk_start: chunk_start + num_frames]
    if len(chunk) < num_frames:
        chunk = chunk + [padded_traj_maps[-1]] * (num_frames - len(chunk))
    return chunk


def load_pipe(model_paths, device):
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

    model_configs = [ModelConfig(path=path) for path in model_paths]
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
    )
    return pipe


def run_episode(pipe, args, ep_key, ep_paths, worker_prefix):
    from diffsynth import save_video

    ep_out_dir = os.path.join(args.output_dir, ep_key)
    images_dir = os.path.join(ep_out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    first_frame, abs_actions, intrinsic_scaled, w2c, total_frames = load_episode_data(
        ep_paths, args.downsample_step, args.H, args.W
    )
    print(f"{worker_prefix} episode={ep_key} total_downsampled_frames={total_frames}", flush=True)

    if total_frames <= 0:
        print(f"{worker_prefix} episode={ep_key} skipped: empty episode", flush=True)
        return

    all_traj_maps = generate_traj_map(
        abs_actions,
        w2c,
        intrinsic_scaled,
        args.H,
        args.W,
        radius=args.traj_radius,
    )

    fixed_num_frames = 1 + args.predict_frames
    total_prediction_frames = max(total_frames - 1, 0)
    num_chunks = math.ceil(total_prediction_frames / args.predict_frames) if total_prediction_frames > 0 else 0

    all_generated_frames = [first_frame]
    first_frame.save(os.path.join(images_dir, "frame_000000.png"))
    current_context = first_frame

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * args.predict_frames
        valid_prediction_count = min(
            args.predict_frames,
            total_prediction_frames - chunk_start,
        )
        control_video = build_fixed_chunk_control_video(
            all_traj_maps=all_traj_maps,
            chunk_start=chunk_start,
            num_frames=fixed_num_frames,
        )

        print(
            f"{worker_prefix} episode={ep_key} "
            f"chunk={chunk_idx + 1}/{num_chunks} "
            f"predict_start={chunk_start + 1} "
            f"valid_prediction_count={valid_prediction_count} "
            f"chunk_num_frames={fixed_num_frames}",
            flush=True,
        )

        generated = pipe(
            prompt=args.prompt,
            control_video=control_video,
            reference_image=current_context,
            height=args.H,
            width=args.W,
            num_frames=fixed_num_frames,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            tiled=True,
        )

        new_frames = generated[1: 1 + valid_prediction_count]
        for i, frame in enumerate(new_frames):
            global_idx = 1 + chunk_start + i
            pil_frame = frame if isinstance(frame, Image.Image) else Image.fromarray(np.array(frame))
            pil_frame.save(os.path.join(images_dir, f"frame_{global_idx:06d}.png"))
            all_generated_frames.append(pil_frame)

        current_context = all_generated_frames[-1]

    if len(all_generated_frames) != total_frames:
        raise RuntimeError(
            f"Episode {ep_key}: generated {len(all_generated_frames)} frames, "
            f"expected {total_frames}."
        )

    video_path = os.path.join(ep_out_dir, "video.mp4")
    save_video(all_generated_frames, video_path, fps=args.fps, quality=5)
    print(f"{worker_prefix} episode={ep_key} saved_frames={len(all_generated_frames)} -> {video_path}", flush=True)


def worker_main(rank, gpu_ids, assignments, args_dict):
    args = argparse.Namespace(**args_dict)
    gpu_id = gpu_ids[rank]
    my_episodes = assignments[rank]
    worker_prefix = f"[GPU {gpu_id}]"

    if not my_episodes:
        print(f"{worker_prefix} no assigned episodes, exiting.", flush=True)
        return

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    print(f"{worker_prefix} loading pipeline for {len(my_episodes)} episode(s)...", flush=True)
    pipe = load_pipe(args.model_paths, device)
    print(f"{worker_prefix} pipeline loaded.", flush=True)

    for ep_key, ep_paths in my_episodes:
        run_episode(pipe, args, ep_key, ep_paths, worker_prefix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", type=str, default="/mnt/workspace/zsq/Agibotsubset/val")
    parser.add_argument("--output_dir", type=str, default="/mnt/workspace/zsq/Agibotsubset/val_results_1.3B_control_5steps")
    parser.add_argument("--original_hz", type=int, default=30)
    parser.add_argument("--target_hz", type=int, default=5)
    parser.add_argument("--predict_frames", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--H", type=int, default=320)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--traj_radius", type=int, default=50)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--prompt", type=str, default="机械臂的两个臂执行动作")
    parser.add_argument("--episode_filter", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument(
        "--model_paths",
        nargs=4,
        default=[
            "/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.1-Fun-1.3B-Control/epoch-49.safetensors",
            "/mnt/workspace/zsq/Wan_model/Wan2.1-Fun-1.3B-Control-deprated/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            "/mnt/workspace/zsq/Wan_model/Wan2.1-Fun-1.3B-Control-deprated/models_t5_umt5-xxl-enc-bf16.pth",
            "/mnt/workspace/zsq/Wan_model/Wan2.1-Fun-1.3B-Control-deprated/Wan2.1_VAE.pth"
        ],
        help="Four model paths used by WanVideoPipeline.from_pretrained.",
    )
    args = parser.parse_args()

    if args.predict_frames <= 0:
        raise ValueError("--predict_frames must be positive.")
    if args.target_hz <= 0 or args.original_hz <= 0:
        raise ValueError("--original_hz and --target_hz must be positive.")
    if args.original_hz % args.target_hz != 0:
        raise ValueError("--original_hz must be divisible by --target_hz.")

    args.downsample_step = args.original_hz // args.target_hz

    episodes = discover_episodes(args.val_path)
    print(f"[INFO] Found {len(episodes)} episode(s) under {args.val_path}")

    if args.episode_filter:
        episodes = [(k, v) for k, v in episodes if k == args.episode_filter]
        print(f"[INFO] Filtered to {len(episodes)} episode(s)")

    if not episodes:
        print("[INFO] No episodes to process.")
        return

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        raise RuntimeError("No GPU ids available. Pass --gpu_ids or expose CUDA devices.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    active_gpu_ids = gpu_ids[: min(len(gpu_ids), len(episodes))]
    assignments = build_episode_assignments(episodes, active_gpu_ids)

    print(f"[INFO] Using GPUs: {active_gpu_ids}")
    for gpu_id, assigned in zip(active_gpu_ids, assignments):
        keys = [k for k, _ in assigned]
        print(f"[INFO] GPU {gpu_id}: {len(keys)} episode(s) -> {keys}")

    os.makedirs(args.output_dir, exist_ok=True)

    mp.spawn(
        worker_main,
        args=(active_gpu_ids, assignments, vars(args)),
        nprocs=len(active_gpu_ids),
        join=True,
    )

    print(f"[DONE] All assigned episodes processed. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
