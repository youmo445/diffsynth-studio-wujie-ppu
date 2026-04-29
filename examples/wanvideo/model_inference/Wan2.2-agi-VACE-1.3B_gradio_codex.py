"""
Gradio demo for interactive multiview Wan VACE generation.

The UI lets you choose an episode/chunk, initializes sliders from the chunk's
context action, repeats one user-controlled action across all 8 predicted
frames, and returns both the generated video and the action-map video.
"""

import argparse
import json
import math
import os
import tempfile
import uuid

import gradio as gr
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


PIPE = None
EPISODE_CACHE = {}

ACTION_FIELDS = [
    ("left_x", 0, -5.0, 5.0),
    ("left_y", 1, -5.0, 5.0),
    ("left_z", 2, -5.0, 5.0),
    ("left_qx", 3, -1.0, 1.0),
    ("left_qy", 4, -1.0, 1.0),
    ("left_qz", 5, -1.0, 1.0),
    ("left_qw", 6, -1.0, 1.0),
    ("left_grip", 7, -10.0, 100.0),
    ("right_x", 8, -5.0, 5.0),
    ("right_y", 9, -5.0, 5.0),
    ("right_z", 10, -5.0, 5.0),
    ("right_qx", 11, -1.0, 1.0),
    ("right_qy", 12, -1.0, 1.0),
    ("right_qz", 13, -1.0, 1.0),
    ("right_qw", 14, -1.0, 1.0),
    ("right_grip", 15, -10.0, 100.0),
]


def discover_episodes(base_path):
    proprio_base = os.path.join(base_path, "proprio_stats")
    episodes = {}
    if not os.path.isdir(proprio_base):
        raise RuntimeError(f"Only split layout is supported in this demo: {base_path}")

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


def get_episode_info(episode_key, args):
    if episode_key not in EPISODE_CACHE:
        episodes = args.episodes
        EPISODE_CACHE[episode_key] = build_episode_info(
            episodes[episode_key],
            args.downsample_step,
            args.camera_names,
        )
    return EPISODE_CACHE[episode_key]


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


def load_multiview_frame(ep_info, camera_names, ds_pos, view_height, view_width):
    orig_idx = int(ep_info["ds_indices"][ds_pos])
    frames = []
    raw_hw = None
    for cam in camera_names:
        video_path = os.path.join(ep_info["video_dir"], f"{cam}_color.mp4")
        frame = read_video_frame(video_path, orig_idx)
        if raw_hw is None:
            raw_hw = frame.shape[:2]
        frame = np.array(Image.fromarray(frame).resize((view_width, view_height), Image.BILINEAR))
        frames.append(frame)
    return Image.fromarray(np.concatenate(frames, axis=0)), raw_hw


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


def pad_camera_seq(seq, start_idx, num_frames):
    chunk = seq[start_idx: start_idx + num_frames]
    if len(chunk) < num_frames:
        pad = [seq[-1]] * (num_frames - len(chunk))
        chunk = list(chunk) + pad
    return np.stack(chunk, axis=0)


def normalize_action_vector(action_vec, ref_action):
    out = np.array(action_vec, dtype=np.float32).copy()
    for start in (3, 11):
        quat = out[start:start + 4]
        norm = float(np.linalg.norm(quat))
        if norm < 1e-6:
            out[start:start + 4] = ref_action[start:start + 4]
        else:
            out[start:start + 4] = quat / norm
    return out


def build_constant_control_sequence(ep_info, start_idx, action_vec, args):
    ref_frame, raw_hw = load_multiview_frame(
        ep_info, args.camera_names, start_idx, args.view_height, args.view_width
    )
    raw_h, raw_w = raw_hw
    repeated_action = np.repeat(action_vec[None, :], args.num_frames, axis=0)

    action_map_per_cam = {}
    ray_map_o = []
    ray_map_d = []

    for cam in args.camera_names:
        cam_info = ep_info["cameras"][cam]
        intrinsic = cam_info["intrinsic"].copy()
        intrinsic[0, 0] *= args.view_width / raw_w
        intrinsic[0, 2] *= args.view_width / raw_w
        intrinsic[1, 1] *= args.view_height / raw_h
        intrinsic[1, 2] *= args.view_height / raw_h

        w2c_seq = pad_camera_seq(cam_info["w2c"], start_idx, args.num_frames)
        action_map_per_cam[cam] = generate_traj_map(
            repeated_action,
            w2c_seq,
            intrinsic,
            args.view_height,
            args.view_width,
            radius=args.traj_radius,
            radius_mode=args.traj_radius_mode,
            min_radius=args.traj_min_radius,
            max_radius=args.traj_max_radius,
            ref_depth=args.traj_ref_depth,
            near_depth=args.traj_near_depth,
            far_depth=args.traj_far_depth,
        )

        if args.output_raymap:
            c2w_seq = pad_camera_seq(cam_info["c2w"], start_idx, args.num_frames)
            ray_o_seq = []
            ray_d_seq = []
            for t in range(args.num_frames):
                ray_o, ray_d = generate_raymap(
                    intrinsic=intrinsic,
                    c2w=c2w_seq[t],
                    height=args.view_height,
                    width=args.view_width,
                    ray_o_vmin=np.array(args.ray_o_vmin, dtype=np.float32),
                    ray_o_vmax=np.array(args.ray_o_vmax, dtype=np.float32),
                    ray_d_vmin=np.array(args.ray_d_vmin, dtype=np.float32),
                    ray_d_vmax=np.array(args.ray_d_vmax, dtype=np.float32),
                )
                ray_o_seq.append(ray_o)
                ray_d_seq.append(ray_d)
            ray_map_o.append(ray_o_seq)
            ray_map_d.append(ray_d_seq)

    control_frames = []
    ray_o_frames = []
    ray_d_frames = []
    for t in range(args.num_frames):
        control_frames.append(
            Image.fromarray(
                np.concatenate([action_map_per_cam[cam][t] for cam in args.camera_names], axis=0)
            )
        )
        if args.output_raymap:
            ray_o_frames.append(
                Image.fromarray(np.concatenate([ray_map_o[i][t] for i, _ in enumerate(args.camera_names)], axis=0))
            )
            ray_d_frames.append(
                Image.fromarray(np.concatenate([ray_map_d[i][t] for i, _ in enumerate(args.camera_names)], axis=0))
            )
    return ref_frame, control_frames, ray_o_frames, ray_d_frames


def save_frames_to_mp4(frames, out_path, fps):
    writer = imageio.get_writer(out_path, fps=fps)
    try:
        for frame in frames:
            writer.append_data(np.array(frame))
    finally:
        writer.close()


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
        pipe.vace.load_state_dict(state_dict)
    return pipe


def get_pipe(args):
    global PIPE
    if PIPE is None:
        PIPE = load_pipe(args.base_model_paths, args.vace_model_path, args.device)
    return PIPE


def chunk_count(ep_info, predict_frames):
    return max(1, int(math.ceil(max(ep_info["T_ds"] - 1, 0) / float(predict_frames))))


def get_context_idx(chunk_id_1based, predict_frames, total_frames):
    context_idx = max(0, (int(chunk_id_1based) - 1) * predict_frames)
    return min(context_idx, max(total_frames - 1, 0))


def preview_from_selection(episode_key, chunk_id, traj_radius_mode, *slider_values):
    args = APP_ARGS
    args.traj_radius_mode = traj_radius_mode
    ep_info = get_episode_info(episode_key, args)
    start_idx = get_context_idx(chunk_id, args.predict_frames, ep_info["T_ds"])
    ref_action = ep_info["abs_actions"][start_idx]

    if slider_values and len(slider_values) == len(ACTION_FIELDS):
        action_vec = normalize_action_vector(np.array(slider_values, dtype=np.float32), ref_action)
    else:
        action_vec = ref_action.copy()

    ref_frame, control_frames, _, _ = build_constant_control_sequence(ep_info, start_idx, action_vec, args)
    preview = control_frames[0]
    info = (
        f"episode={episode_key}\n"
        f"chunk={chunk_id}/{chunk_count(ep_info, args.predict_frames)}\n"
        f"context_idx={start_idx}\n"
        f"traj_radius_mode={traj_radius_mode}\n"
        f"predict_frames={args.predict_frames}\n"
        f"action_repeated_frames={args.predict_frames}"
    )
    return np.array(ref_frame), np.array(preview), info


def load_chunk_defaults(episode_key, chunk_id):
    args = APP_ARGS
    ep_info = get_episode_info(episode_key, args)
    start_idx = get_context_idx(chunk_id, args.predict_frames, ep_info["T_ds"])
    action_vec = ep_info["abs_actions"][start_idx].astype(np.float32)
    ref_frame, preview, info = preview_from_selection(
        episode_key,
        chunk_id,
        args.traj_radius_mode,
        *action_vec.tolist(),
    )

    slider_updates = [gr.update(value=float(action_vec[idx])) for _, idx, _, _ in ACTION_FIELDS]
    chunk_update = gr.update(
        minimum=1,
        maximum=chunk_count(ep_info, args.predict_frames),
        step=1,
        value=min(chunk_id, chunk_count(ep_info, args.predict_frames)),
    )
    return [chunk_update] + slider_updates + [ref_frame, preview, info]


def run_generation(episode_key, chunk_id, traj_radius_mode, seed, num_inference_steps, cfg_scale, *slider_values):
    args = APP_ARGS
    args.traj_radius_mode = traj_radius_mode
    ep_info = get_episode_info(episode_key, args)
    start_idx = get_context_idx(chunk_id, args.predict_frames, ep_info["T_ds"])
    ref_action = ep_info["abs_actions"][start_idx]
    action_vec = normalize_action_vector(np.array(slider_values, dtype=np.float32), ref_action)

    ref_frame, control_frames, ray_o_frames, ray_d_frames = build_constant_control_sequence(
        ep_info, start_idx, action_vec, args
    )

    pipe = get_pipe(args)
    pipe_kwargs = {
        "prompt": args.prompt,
        "vace_video": control_frames,
        "vace_reference_image": ref_frame,
        "height": args.view_height * len(args.camera_names),
        "width": args.view_width,
        "num_frames": args.num_frames,
        "num_inference_steps": int(num_inference_steps),
        "cfg_scale": float(cfg_scale),
        "seed": int(seed),
        "tiled": False,
    }
    if args.output_raymap:
        pipe_kwargs["ray_map_o"] = ray_o_frames
        pipe_kwargs["ray_map_d"] = ray_d_frames

    generated = pipe(**pipe_kwargs)
    generated = [frame if isinstance(frame, Image.Image) else Image.fromarray(np.array(frame)) for frame in generated]

    out_dir = os.path.join(tempfile.gettempdir(), "wan_gradio_codex")
    os.makedirs(out_dir, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    gen_path = os.path.join(out_dir, f"{episode_key}_chunk{chunk_id}_{traj_radius_mode}_{token}_gen.mp4")
    act_path = os.path.join(out_dir, f"{episode_key}_chunk{chunk_id}_{traj_radius_mode}_{token}_action.mp4")
    save_frames_to_mp4(generated, gen_path, args.fps)
    save_frames_to_mp4(control_frames, act_path, args.fps)

    status = (
        f"Generated {len(generated)} frames.\n"
        f"episode={episode_key}\n"
        f"chunk={chunk_id}/{chunk_count(ep_info, args.predict_frames)}\n"
        f"context_idx={start_idx}\n"
        f"traj_radius_mode={traj_radius_mode}\n"
        f"output_raymap={args.output_raymap}\n"
        f"seed={seed}"
    )
    return gen_path, act_path, status


def build_ui(args):
    episode_keys = sorted(args.episodes.keys())
    default_episode = args.default_episode or episode_keys[0]
    ep_info = get_episode_info(default_episode, args)
    default_chunk = max(1, min(args.default_chunk, chunk_count(ep_info, args.predict_frames)))
    default_action = ep_info["abs_actions"][get_context_idx(default_chunk, args.predict_frames, ep_info["T_ds"])]

    with gr.Blocks(title="Wan Multiview World Model Gradio Codex") as demo:
        gr.Markdown(
            "# Wan Multiview World Model\n"
            "选择 episode 和 chunk，滑动动作条，使用同一个动作重复控制 8 个预测帧，并同时导出生成视频和 action_map 视频。"
        )
        with gr.Row():
            episode_dropdown = gr.Dropdown(
                choices=episode_keys,
                value=default_episode,
                label="Episode",
            )
            chunk_slider = gr.Slider(
                minimum=1,
                maximum=chunk_count(ep_info, args.predict_frames),
                step=1,
                value=default_chunk,
                label="Chunk (1-based)",
            )
            traj_mode = gr.Radio(
                choices=["constant", "perspective", "depth_norm"],
                value=args.traj_radius_mode,
                label="Traj Radius Mode",
            )
            load_button = gr.Button("Load Chunk Defaults", variant="secondary")

        with gr.Row():
            seed_slider = gr.Slider(minimum=0, maximum=1000000, step=1, value=args.seed, label="Seed")
            step_slider = gr.Slider(
                minimum=1, maximum=100, step=1, value=args.num_inference_steps, label="Inference Steps"
            )
            cfg_slider = gr.Slider(minimum=1.0, maximum=12.0, step=0.1, value=args.cfg_scale, label="CFG Scale")
            run_button = gr.Button("Generate", variant="primary")

        slider_components = []
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Left Arm")
                for name, idx, vmin, vmax in ACTION_FIELDS[:8]:
                    slider_components.append(
                        gr.Slider(minimum=vmin, maximum=vmax, step=0.001, value=float(default_action[idx]), label=name)
                    )
            with gr.Column():
                gr.Markdown("## Right Arm")
                for name, idx, vmin, vmax in ACTION_FIELDS[8:]:
                    slider_components.append(
                        gr.Slider(minimum=vmin, maximum=vmax, step=0.001, value=float(default_action[idx]), label=name)
                    )

        with gr.Row():
            ref_image = gr.Image(label="Reference Multiview Frame", type="numpy")
            map_preview = gr.Image(label="Action Map Preview (frame 0)", type="numpy")
        status_box = gr.Textbox(label="Status", lines=8)

        with gr.Row():
            gen_video = gr.Video(label="Generated Video")
            action_video = gr.Video(label="Action Map Video")

        load_outputs = [chunk_slider] + slider_components + [ref_image, map_preview, status_box]
        load_inputs = [episode_dropdown, chunk_slider]
        load_button.click(load_chunk_defaults, inputs=load_inputs, outputs=load_outputs)
        episode_dropdown.change(load_chunk_defaults, inputs=load_inputs, outputs=load_outputs)

        preview_inputs = [episode_dropdown, chunk_slider, traj_mode] + slider_components
        preview_outputs = [ref_image, map_preview, status_box]
        chunk_slider.change(preview_from_selection, inputs=preview_inputs, outputs=preview_outputs)
        traj_mode.change(preview_from_selection, inputs=preview_inputs, outputs=preview_outputs)
        for slider in slider_components:
            slider.release(preview_from_selection, inputs=preview_inputs, outputs=preview_outputs)

        run_button.click(
            run_generation,
            inputs=[episode_dropdown, chunk_slider, traj_mode, seed_slider, step_slider, cfg_slider] + slider_components,
            outputs=[gen_video, action_video, status_box],
        )

        initial_ref, initial_map, initial_info = preview_from_selection(
            default_episode, default_chunk, args.traj_radius_mode, *default_action.tolist()
        )
        demo.load(
            lambda: (initial_ref, initial_map, initial_info),
            outputs=[ref_image, map_preview, status_box],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", type=str, default="/mnt/workspace/zsq/Agi2024subset_split/val")
    parser.add_argument(
        "--base_model_paths",
        type=str,
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
        default="/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.1-VACE-1.3B-multiview-raymap-perspectivate/epoch-59.safetensors",
    )
    parser.add_argument("--output_dir", type=str, default="/mnt/workspace/zsq/DiffSynth-Studio/outputs/gradio_codex")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--camera_names", nargs="+", default=["head", "hand_left", "hand_right"])
    parser.add_argument("--orig_height", type=int, default=480)
    parser.add_argument("--orig_width", type=int, default=640)
    parser.add_argument("--view_height", type=int, default=480)
    parser.add_argument("--view_width", type=int, default=640)
    parser.add_argument("--original_hz", type=int, default=30)
    parser.add_argument("--target_hz", type=int, default=5)
    parser.add_argument("--predict_frames", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=9)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--default_episode", type=str, default=None)
    parser.add_argument("--default_chunk", type=int, default=1)
    parser.add_argument("--traj_radius", type=int, default=3)
    parser.add_argument("--traj_radius_mode", type=str, default="perspective", choices=["constant", "perspective", "depth_norm"])
    parser.add_argument("--traj_min_radius", type=float, default=2.0)
    parser.add_argument("--traj_max_radius", type=float, default=7.0)
    parser.add_argument("--traj_ref_depth", type=float, default=1.0)
    parser.add_argument("--traj_near_depth", type=float, default=0.3)
    parser.add_argument("--traj_far_depth", type=float, default=2.0)
    parser.add_argument("--output_raymap", action="store_true")
    parser.add_argument("--ray_o_vmin", type=float, nargs=3, default=[-1.5, -1.5, -1.5])
    parser.add_argument("--ray_o_vmax", type=float, nargs=3, default=[1.5, 1.5, 1.5])
    parser.add_argument("--ray_d_vmin", type=float, nargs=3, default=[-1.0, -1.0, -1.0])
    parser.add_argument("--ray_d_vmax", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    args = parser.parse_args()
    args.downsample_step = max(1, args.original_hz // args.target_hz)
    return args


if __name__ == "__main__":
    APP_ARGS = parse_args()
    APP_ARGS.episodes = discover_episodes(APP_ARGS.val_path)
    os.makedirs(APP_ARGS.output_dir, exist_ok=True)
    demo = build_ui(APP_ARGS)
    demo.queue().launch(
        server_name=APP_ARGS.server_name,
        server_port=APP_ARGS.server_port,
        share=APP_ARGS.share,
    )
