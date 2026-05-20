"""
Gradio online inference for Wan2.2 TI2V-5B + VACE (ARXX5).

Features:
- Choose an ARXX5 episode/chunk as initialization.
- Action textbox is auto-filled with the selected init action (16 dims: left8 + right8, including grippers).
- For one chunk, the future 8 frames use the same user action.
- Export both control video and generated future video.
"""

import argparse
import json
import math
import os
import re
import tempfile
import uuid
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

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


PIPE = None
EPISODE_INFO_CACHE = {}
APP_ARGS = None


def discover_episodes(agx_base, camera_names):
    base = Path(agx_base)
    data_dir = base / "data"
    videos_dir = base / "videos"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    episodes = {}
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
            key = f"episode_{ep_idx:06d}@{chunk}"
            episodes[key] = {
                "episode_index": ep_idx,
                "chunk": chunk,
                "parquet_path": parquet_path,
                "video_paths": video_paths,
            }
    if not episodes:
        raise RuntimeError(f"No valid episodes found in {agx_base}")
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


def get_episode_info(episode_key):
    if episode_key not in EPISODE_INFO_CACHE:
        EPISODE_INFO_CACHE[episode_key] = build_episode_info(APP_ARGS.episodes[episode_key], APP_ARGS)
    return EPISODE_INFO_CACHE[episode_key]


def chunk_count(ep_info):
    total_prediction_frames = max(ep_info["T_ds"] - 1, 0)
    return max(1, int(math.ceil(total_prediction_frames / float(APP_ARGS.predict_frames))))


def get_context_idx(chunk_id_1based, ep_info):
    context_idx = max(0, (int(chunk_id_1based) - 1) * APP_ARGS.predict_frames)
    return min(context_idx, max(ep_info["T_ds"] - 1, 0))


def format_action16(action16):
    vals = list(action16[0:8]) + list(action16[8:16])
    return ", ".join([f"{float(v):.6f}" for v in vals])


def normalize_xyzw_quat(v):
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return v
    return v / norm


def parse_action_text(action_text, ref_action):
    tokens = action_text.replace("\n", " ").replace(";", " ").replace(",", " ").split()
    vals = [float(x) for x in tokens]

    out = ref_action.copy().astype(np.float32)
    if len(vals) == 14:
        out[0:7] = np.asarray(vals[0:7], dtype=np.float32)
        out[8:15] = np.asarray(vals[7:14], dtype=np.float32)
    elif len(vals) == 16:
        out[:] = np.asarray(vals, dtype=np.float32)
    else:
        raise ValueError(f"Action dim must be 14 or 16, got {len(vals)}")

    out[3:7] = normalize_xyzw_quat(out[3:7])
    out[11:15] = normalize_xyzw_quat(out[11:15])
    return out


def load_ref_multiview_frame(ep_info, ds_pos):
    raw_idx = int(ep_info["ds_indices"][ds_pos])
    frames = []
    raw_hw = None
    for cam in APP_ARGS.camera_names:
        frame = read_video_rgb_by_indices(ep_info["video_paths"][cam], [raw_idx])[0]
        if raw_hw is None:
            raw_hw = frame.shape[:2]
        frame = np.array(Image.fromarray(frame).resize((APP_ARGS.view_width, APP_ARGS.view_height), Image.BILINEAR))
        frames.append(frame)
    return Image.fromarray(np.concatenate(frames, axis=0)), raw_hw


def get_chunk_with_pad_np(seq, start_idx, num_frames):
    chunk = seq[start_idx: start_idx + num_frames]
    if len(chunk) < num_frames:
        chunk = list(chunk) + [seq[-1]] * (num_frames - len(chunk))
    return np.stack(chunk, axis=0)


def build_chunk_controls(ep_info, chunk_id, action_text):
    start_idx = get_context_idx(chunk_id, ep_info)
    ref_action = ep_info["traj_actions_by_cam"][APP_ARGS.camera_names[0]][start_idx]
    user_action = parse_action_text(action_text, ref_action)

    action_seq = np.repeat(user_action[None, :], APP_ARGS.num_frames, axis=0)
    action_seq[0] = ref_action

    ref_frame, raw_hw = load_ref_multiview_frame(ep_info, start_idx)
    raw_h, raw_w = raw_hw

    intrinsics = get_color_intrinsics()
    control_per_cam = {}
    ray_o_per_cam = {}
    ray_d_per_cam = {}

    for cam in APP_ARGS.camera_names:
        k = scale_intrinsic(intrinsics[cam], raw_h, raw_w, APP_ARGS.view_height, APP_ARGS.view_width)

        traj_w2c_full = ep_info["traj_w2c_by_cam"][cam]
        traj_w2c = get_chunk_with_pad_np(traj_w2c_full, start_idx, APP_ARGS.num_frames)

        control_per_cam[cam] = generate_traj_map(
            action_seq,
            traj_w2c,
            k,
            APP_ARGS.view_height,
            APP_ARGS.view_width,
            radius=APP_ARGS.traj_radius,
            radius_mode=APP_ARGS.traj_radius_mode,
            min_radius=APP_ARGS.traj_min_radius,
            max_radius=APP_ARGS.traj_max_radius,
            ref_depth=APP_ARGS.traj_ref_depth,
            near_depth=APP_ARGS.traj_near_depth,
            far_depth=APP_ARGS.traj_far_depth,
        )

        ray_w2c_full = ep_info["ray_w2c_by_cam"][cam]
        ray_w2c = get_chunk_with_pad_np(ray_w2c_full, start_idx, APP_ARGS.num_frames)
        ray_o_seq = []
        ray_d_seq = []
        for t in range(APP_ARGS.num_frames):
            c2w = np.linalg.inv(ray_w2c[t]).astype(np.float32)
            ro, rd = generate_raymap(k, c2w, APP_ARGS.view_height, APP_ARGS.view_width)
            ray_o_seq.append(to_uint8_img(ro, APP_ARGS.ray_o_vmin, APP_ARGS.ray_o_vmax))
            ray_d_seq.append(to_uint8_img(rd, APP_ARGS.ray_d_vmin, APP_ARGS.ray_d_vmax))
        ray_o_per_cam[cam] = ray_o_seq
        ray_d_per_cam[cam] = ray_d_seq

    control_video = []
    ray_map_o = []
    ray_map_d = []
    for t in range(APP_ARGS.num_frames):
        control_video.append(
            Image.fromarray(np.concatenate([control_per_cam[cam][t] for cam in APP_ARGS.camera_names], axis=0))
        )
        ray_map_o.append(
            Image.fromarray(np.concatenate([ray_o_per_cam[cam][t] for cam in APP_ARGS.camera_names], axis=0))
        )
        ray_map_d.append(
            Image.fromarray(np.concatenate([ray_d_per_cam[cam][t] for cam in APP_ARGS.camera_names], axis=0))
        )

    return ref_frame, control_video, ray_map_o, ray_map_d, format_action16(user_action), start_idx


def save_frames_to_mp4(frames, path, fps):
    import imageio.v2 as imageio

    writer = imageio.get_writer(path, fps=fps)
    try:
        for frame in frames:
            writer.append_data(np.array(frame))
    finally:
        writer.close()


def load_pipe(base_model_paths, vace_model_path, device, vace_in_dim=144):
    import torch

    from diffsynth.models.utils import load_state_dict
    from diffsynth.models.wan_video_vace_wan22_codex import VaceWan22CodexModel
    from diffsynth.pipelines.wan_video_new_wan22_codex import ModelConfig, WanVideoPipeline

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


def get_pipe():
    global PIPE
    if PIPE is None:
        PIPE = load_pipe(APP_ARGS.base_model_paths, APP_ARGS.vace_model_path, APP_ARGS.device, APP_ARGS.vace_in_dim)
    return PIPE


def on_episode_or_chunk_change(episode_key, chunk_id):
    ep_info = get_episode_info(episode_key)
    max_chunk = chunk_count(ep_info)
    chunk_id = int(max(1, min(int(chunk_id), max_chunk)))
    start_idx = get_context_idx(chunk_id, ep_info)

    init_action = ep_info["traj_actions_by_cam"][APP_ARGS.camera_names[0]][start_idx]
    action_text = format_action16(init_action)

    ref_frame, control_video, _, _, _, _ = build_chunk_controls(ep_info, chunk_id, action_text)
    status = (
        f"episode={episode_key}\n"
        f"chunk={chunk_id}/{max_chunk}\n"
        f"context_idx={start_idx}\n"
        f"future_frames={APP_ARGS.predict_frames}\n"
        "action format: 16 dims = left8 + right8 (with gripper); 14 dims still supported"
    )

    return (
        gr.update(minimum=1, maximum=max_chunk, step=1, value=chunk_id),
        action_text,
        np.array(ref_frame),
        np.array(control_video[1]),
        status,
    )


def run_generation(episode_key, chunk_id, action_text, seed, steps, cfg_scale):
    ep_info = get_episode_info(episode_key)
    max_chunk = chunk_count(ep_info)
    chunk_id = int(max(1, min(int(chunk_id), max_chunk)))

    ref_frame, control_video, ray_map_o, ray_map_d, normalized_action, start_idx = build_chunk_controls(
        ep_info, chunk_id, action_text
    )

    pipe = get_pipe()
    total_h = APP_ARGS.view_height * len(APP_ARGS.camera_names)
    generated = pipe(
        prompt=APP_ARGS.prompt,
        input_image=ref_frame,
        vace_video=control_video,
        ray_map_o=ray_map_o,
        ray_map_d=ray_map_d,
        height=total_h,
        width=APP_ARGS.view_width,
        num_frames=APP_ARGS.num_frames,
        num_inference_steps=int(steps),
        cfg_scale=float(cfg_scale),
        seed=int(seed),
        tiled=APP_ARGS.tiled,
    )
    generated = [x if isinstance(x, Image.Image) else Image.fromarray(np.array(x)) for x in generated]

    control_future = control_video[1:]
    generated_future = generated[1:]

    out_dir = os.path.join(tempfile.gettempdir(), "wan22_arxx5_gradio")
    os.makedirs(out_dir, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    control_path = os.path.join(out_dir, f"{token}_control_future.mp4")
    gen_path = os.path.join(out_dir, f"{token}_generated_future.mp4")

    save_frames_to_mp4(control_future, control_path, APP_ARGS.fps)
    save_frames_to_mp4(generated_future, gen_path, APP_ARGS.fps)

    status = (
        f"done\n"
        f"episode={episode_key}\n"
        f"chunk={chunk_id}/{max_chunk}\n"
        f"context_idx={start_idx}\n"
        f"future_generated_frames={len(generated_future)}\n"
        f"seed={seed}, steps={steps}, cfg={cfg_scale}"
    )

    return control_path, gen_path, normalized_action, status


def build_demo():
    episode_keys = sorted(APP_ARGS.episodes.keys())
    default_episode = episode_keys[0]
    ep_info = get_episode_info(default_episode)
    default_chunk = 1
    init_action = ep_info["traj_actions_by_cam"][APP_ARGS.camera_names[0]][0]
    default_action_text = format_action16(init_action)

    with gr.Blocks(title="Wan2.2 ARXX5 Online Inference") as demo:
        gr.Markdown(
            "# Wan2.2 TI2V-5B + VACE (ARXX5) Online Inference\n"
            "输入 16 维动作（left8 + right8，包含两个 gripper），未来 8 帧会重复该动作。"
        )

        with gr.Row():
            episode_dropdown = gr.Dropdown(choices=episode_keys, value=default_episode, label="Episode")
            chunk_slider = gr.Slider(minimum=1, maximum=chunk_count(ep_info), step=1, value=default_chunk, label="Chunk")

        action_text = gr.Textbox(
            label="Action (16 dims: left8 + right8, includes grippers)",
            value=default_action_text,
            lines=3,
        )

        with gr.Row():
            seed_slider = gr.Slider(minimum=0, maximum=1000000, step=1, value=APP_ARGS.seed, label="Seed")
            step_slider = gr.Slider(minimum=1, maximum=100, step=1, value=APP_ARGS.num_inference_steps, label="Steps")
            cfg_slider = gr.Slider(minimum=1.0, maximum=12.0, step=0.1, value=APP_ARGS.cfg_scale, label="CFG")

        with gr.Row():
            load_btn = gr.Button("Load Init Action", variant="secondary")
            run_btn = gr.Button("Generate 8 Future Frames", variant="primary")

        with gr.Row():
            ref_image = gr.Image(label="Reference Frame", type="numpy")
            control_preview = gr.Image(label="Control Preview (future frame 1)", type="numpy")

        with gr.Row():
            control_video = gr.Video(label="Control Video (future 8 frames)")
            gen_video = gr.Video(label="Generated Video (future 8 frames)")

        status = gr.Textbox(label="Status", lines=8)

        load_outputs = [chunk_slider, action_text, ref_image, control_preview, status]
        load_inputs = [episode_dropdown, chunk_slider]
        load_btn.click(on_episode_or_chunk_change, inputs=load_inputs, outputs=load_outputs)
        episode_dropdown.change(on_episode_or_chunk_change, inputs=load_inputs, outputs=load_outputs)
        chunk_slider.change(on_episode_or_chunk_change, inputs=load_inputs, outputs=load_outputs)

        run_btn.click(
            run_generation,
            inputs=[episode_dropdown, chunk_slider, action_text, seed_slider, step_slider, cfg_slider],
            outputs=[control_video, gen_video, action_text, status],
        )

        init_vals = on_episode_or_chunk_change(default_episode, default_chunk)
        demo.load(
            lambda: init_vals,
            outputs=load_outputs,
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--agx_base", type=str, default="/mnt/data/zsq/agx")
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

    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vace_in_dim", type=int, default=144)

    parser.add_argument("--head_left_calib", type=str, default="/mnt/data/zsq/outputs/calib_eye_to_hand_head_left/result_eye_to_hand.json")
    parser.add_argument("--head_right_calib", type=str, default="/mnt/data/zsq/outputs/calib_eye_to_hand_head_right/result_eye_to_hand.json")
    parser.add_argument("--left_calib", type=str, default="/mnt/data/zsq/outputs/calib_eye_in_hand_left/result_eye_in_hand.json")
    parser.add_argument("--right_calib", type=str, default="/mnt/data/zsq/outputs/calib_eye_in_hand_right/result_eye_in_hand.json")

    parser.add_argument("--base_model_paths", type=str, default=json.dumps([
        [
            "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
            "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
            "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors",
        ],
        "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
        "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
    ]))
    parser.add_argument(
        "--vace_model_path",
        type=str,
        default="/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.2-TI2V-5B-VACE-wan22-codex-ray-action-perspective-arxx5/epoch-19.safetensors",
    )

    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")

    args = parser.parse_args()
    if args.original_hz % args.target_hz != 0:
        raise ValueError("--original_hz must be divisible by --target_hz.")
    if args.num_frames != args.predict_frames + 1:
        raise ValueError("--num_frames must equal --predict_frames + 1.")

    args.downsample_step = args.original_hz // args.target_hz
    args.base_model_paths = json.loads(args.base_model_paths)
    return args


if __name__ == "__main__":
    APP_ARGS = parse_args()
    APP_ARGS.episodes = discover_episodes(APP_ARGS.agx_base, APP_ARGS.camera_names)
    demo = build_demo()
    demo.queue().launch(
        server_name=APP_ARGS.server_name,
        server_port=APP_ARGS.server_port,
        share=APP_ARGS.share,
    )
