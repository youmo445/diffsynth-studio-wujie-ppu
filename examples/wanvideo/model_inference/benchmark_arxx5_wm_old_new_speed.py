#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    import imageio.v2 as imageio
except Exception as exc:
    raise ImportError("This benchmark needs imageio to save mp4 videos.") from exc

try:
    import pyarrow.parquet as pq
except Exception as exc:
    raise ImportError("This benchmark needs pyarrow to read LeRobot parquet files.") from exc

try:
    from decord import VideoReader, cpu
except Exception:
    VideoReader = None
    cpu = None


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.models.utils import load_state_dict
from diffsynth.models.wan_video_vace_wan22_codex import VaceWan22CodexModel
from diffsynth.pipelines.wan_video_new_wan22_codex import ModelConfig, WanVideoPipeline
from diffsynth.trainers.arxx5_dataset4_wancontrolmultiview import (
    configure_fk,
    generate_raymap,
    generate_traj_map,
    get_color_intrinsics,
    load_calib_extrinsics,
    make_center_ray_w2c_sequences,
    make_ee_pose_from_action,
    make_per_camera_sequences,
    read_video_rgb_by_indices,
    scale_intrinsic,
    to_uint8_img,
)
from diffsynth.utils import PipelineUnit


CAMERA_VIDEO_KEY_CANDIDATES = {
    "head": ("head", "base_0_rgb"),
    "left_wrist": ("left_wrist", "left_wrist_0_rgb"),
    "right_wrist": ("right_wrist", "right_wrist_0_rgb"),
}


class WanVideoUnitTI2V2Context5(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "latents", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",),
        )

    def process(self, pipe, input_video, noise, latents, tiled, tile_size, tile_stride):
        if input_video is None:
            return {}
        if len(input_video) != 5:
            raise ValueError(f"Expected 5 context frames, got {len(input_video)}.")
        pipe.load_models_to_device(["vae"])
        context = pipe.preprocess_video(input_video)
        context_latents = pipe.vae.encode(
            context,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if context_latents.shape[2] < 2:
            raise ValueError(f"5 context frames should encode to at least 2 latent slots, got {context_latents.shape}.")
        if latents is None:
            latents = noise.clone()
        latents[:, :, 0:2] = context_latents[:, :, 0:2]
        return {
            "latents": latents,
            "fuse_vae_embedding_in_latents": True,
            "clean_latent_slots": 2,
            "first_frame_latents": context_latents[:, :, 0:2],
        }


class StageTimer:
    def __init__(self, pipe, device: str):
        self.pipe = pipe
        self.device = device
        self.records: List[dict] = []
        self.current: Optional[dict] = None
        self._orig_encode = None
        self._orig_decode = None
        self._orig_model_fn = None

    def _sync(self):
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def __enter__(self):
        self._orig_encode = self.pipe.vae.encode
        self._orig_decode = self.pipe.vae.decode
        self._orig_model_fn = self.pipe.model_fn

        def timed_encode(*args, **kwargs):
            self._sync()
            t0 = time.perf_counter()
            out = self._orig_encode(*args, **kwargs)
            self._sync()
            if self.current is not None:
                self.current["vae_encode_s"] += time.perf_counter() - t0
                self.current["vae_encode_calls"] += 1
            return out

        def timed_decode(*args, **kwargs):
            self._sync()
            t0 = time.perf_counter()
            out = self._orig_decode(*args, **kwargs)
            self._sync()
            if self.current is not None:
                self.current["vae_decode_s"] += time.perf_counter() - t0
                self.current["vae_decode_calls"] += 1
            return out

        def timed_model_fn(*args, **kwargs):
            self._sync()
            t0 = time.perf_counter()
            out = self._orig_model_fn(*args, **kwargs)
            self._sync()
            if self.current is not None:
                self.current["dit_s"] += time.perf_counter() - t0
                self.current["dit_calls"] += 1
            return out

        self.pipe.vae.encode = timed_encode
        self.pipe.vae.decode = timed_decode
        self.pipe.model_fn = timed_model_fn
        return self

    def __exit__(self, exc_type, exc, tb):
        self.pipe.vae.encode = self._orig_encode
        self.pipe.vae.decode = self._orig_decode
        self.pipe.model_fn = self._orig_model_fn

    def start_chunk(self, experiment: str, chunk_idx: int):
        self.current = {
            "experiment": experiment,
            "chunk_idx": int(chunk_idx),
            "total_s": 0.0,
            "vae_encode_s": 0.0,
            "vae_encode_calls": 0,
            "vae_decode_s": 0.0,
            "vae_decode_calls": 0,
            "dit_s": 0.0,
            "dit_calls": 0,
            "other_s": 0.0,
        }

    def end_chunk(self, total_s: float, valid_frames: int):
        if self.current is None:
            return
        self.current["total_s"] = float(total_s)
        self.current["valid_frames"] = int(valid_frames)
        known = self.current["vae_encode_s"] + self.current["vae_decode_s"] + self.current["dit_s"]
        self.current["other_s"] = float(total_s - known)
        self.current["chunk_fps"] = float(valid_frames / max(total_s, 1e-8))
        self.records.append(self.current)
        self.current = None


def split_dit_vace_state_dict(state_dict):
    dit_state, vace_state, unknown_keys = {}, {}, []
    for name, value in state_dict.items():
        if name.startswith("pipe.dit."):
            dit_state[name[len("pipe.dit."):]] = value
        elif name.startswith("dit."):
            dit_state[name[len("dit."):]] = value
        elif name.startswith("pipe.vace."):
            vace_state[name[len("pipe.vace."):]] = value
        elif name.startswith("vace."):
            vace_state[name[len("vace."):]] = value
        else:
            unknown_keys.append(name)
    return dit_state, vace_state, unknown_keys


def load_pipe(args, vace_model_path: str, vace_in_dim: int, view_embedding_num_views: int):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[ModelConfig(path=path) for path in args.base_model_paths],
    )
    pipe.units = [
        WanVideoUnitTI2V2Context5() if unit.__class__.__name__ == "WanVideoUnit_InputVideoEmbedder" else unit
        for unit in pipe.units
    ]
    pipe.vace = VaceWan22CodexModel(
        vace_in_dim=vace_in_dim,
        view_embedding_num_views=view_embedding_num_views,
    ).to(dtype=pipe.torch_dtype, device=pipe.device)

    state_dict = load_state_dict(vace_model_path)
    dit_state, vace_state, unknown_keys = split_dit_vace_state_dict(state_dict)
    if dit_state or vace_state:
        if dit_state:
            incompatible = pipe.dit.load_state_dict(dit_state, strict=False)
            print(
                f"[load] DiT {Path(vace_model_path).name}: keys={len(dit_state)} "
                f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
            )
        if vace_state:
            if any(name.startswith("vace_global_") for name in vace_state):
                pipe.vace.enable_global_cross_attn(global_context_dim=16)
            incompatible = pipe.vace.load_state_dict(vace_state, strict=False)
            print(
                f"[load] VACE {Path(vace_model_path).name}: keys={len(vace_state)} "
                f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
            )
        if unknown_keys:
            print(f"[load] ignored non-DiT/VACE keys={len(unknown_keys)}")
    else:
        if any(name.startswith("vace_global_") for name in state_dict):
            pipe.vace.enable_global_cross_attn(global_context_dim=16)
        pipe.vace.load_state_dict(state_dict)
        print(f"[load] pure VACE {Path(vace_model_path).name}: keys={len(state_dict)}")
    return pipe


def get_video_len(path: Path) -> int:
    if VideoReader is not None:
        return len(VideoReader(str(path), ctx=cpu(0)))
    try:
        import cv2
    except Exception as exc:
        raise ImportError("Need decord or cv2 to count video frames.") from exc
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def find_video_path(base_path: Path, chunk: str, camera_name: str, episode_index: int) -> Path:
    for key in CAMERA_VIDEO_KEY_CANDIDATES.get(camera_name, (camera_name,)):
        path = base_path / "videos" / chunk / f"observation.images.{key}" / f"episode_{episode_index:06d}.mp4"
        if path.exists():
            return path
    return base_path / "videos" / chunk / f"observation.images.{camera_name}" / f"episode_{episode_index:06d}.mp4"


def discover_episodes(base_path: Path, camera_names: Sequence[str]) -> List[dict]:
    episodes = []
    for parquet_path in sorted((base_path / "data").glob("chunk-*/episode_*.parquet")):
        match = re.search(r"episode_(\d+)\.parquet$", parquet_path.name)
        if not match:
            continue
        episode_index = int(match.group(1))
        chunk = parquet_path.parent.name
        video_paths = {
            camera: find_video_path(base_path, chunk, camera, episode_index)
            for camera in camera_names
        }
        if all(path.exists() for path in video_paths.values()):
            episodes.append(
                {
                    "episode_index": episode_index,
                    "chunk": chunk,
                    "parquet_path": parquet_path,
                    "video_paths": video_paths,
                }
            )
    if not episodes:
        raise RuntimeError(f"No usable episodes found in {base_path}")
    return episodes


def select_episode(episodes: Sequence[dict], episode_index: Optional[int], episode_pos: int) -> dict:
    if episode_index is not None:
        for ep in episodes:
            if ep["episode_index"] == episode_index:
                return ep
        raise ValueError(f"episode_index={episode_index} not found.")
    if episode_pos < 0 or episode_pos >= len(episodes):
        raise ValueError(f"episode_pos={episode_pos} out of range [0, {len(episodes) - 1}].")
    return episodes[episode_pos]


def load_episode_action(parquet_path: Path) -> np.ndarray:
    table = pq.read_table(parquet_path)
    for key in ("action", "actions", "observation.state", "state"):
        if key in table.column_names:
            values = table[key].to_pylist()
            return np.asarray(values, dtype=np.float32)
    raise KeyError(f"No action-like column found in {parquet_path}. columns={table.column_names}")


def build_episode_info(ep: dict, args) -> dict:
    action = load_episode_action(ep["parquet_path"])
    if action.ndim != 2:
        action = action.reshape(action.shape[0], -1)
    if action.shape[1] > 14:
        action = action[:, :14]
    if action.shape[1] != 14:
        raise ValueError(f"Expected action dim 14, got {action.shape} from {ep['parquet_path']}")

    n = action.shape[0]
    for camera in args.camera_names:
        n = min(n, get_video_len(ep["video_paths"][camera]))
    raw_indices = np.arange(0, n, args.downsample_step, dtype=np.int64)
    needed_frames = 1 + args.num_chunks * args.horizon_frames
    if len(raw_indices) < needed_frames:
        raise ValueError(
            f"Episode has only {len(raw_indices)} downsampled frames, but benchmark needs {needed_frames}. "
            "Choose a longer episode or reduce --num_chunks."
        )
    raw_indices = raw_indices[:needed_frames]
    action_ds = action[raw_indices]

    fk_ee_pose_ds = make_ee_pose_from_action(action_ds)
    extr = load_calib_extrinsics(
        Path(args.head_left_calib),
        Path(args.head_right_calib),
        Path(args.left_calib),
        Path(args.right_calib),
    )
    traj_actions_by_cam, traj_w2c_by_cam = make_per_camera_sequences(
        fk_ee_pose_ds,
        action_ds,
        extr,
        args.camera_axis_mode,
    )
    ray_w2c_by_cam = make_center_ray_w2c_sequences(
        fk_ee_pose_ds,
        extr,
        args.camera_axis_mode,
    )
    return {
        "episode_index": ep["episode_index"],
        "chunk": ep["chunk"],
        "parquet_path": ep["parquet_path"],
        "video_paths": ep["video_paths"],
        "ds_indices": raw_indices,
        "T_ds": len(raw_indices),
        "action_ds": action_ds,
        "traj_actions_by_cam": traj_actions_by_cam,
        "traj_w2c_by_cam": traj_w2c_by_cam,
        "ray_w2c_by_cam": ray_w2c_by_cam,
    }


def load_first_multiview_frame(ep_info: dict, camera_names: Sequence[str], out_h: int, out_w: int) -> Image.Image:
    raw_idx = int(ep_info["ds_indices"][0])
    frames = []
    for camera in camera_names:
        frame = read_video_rgb_by_indices(ep_info["video_paths"][camera], [raw_idx])[0]
        frame = np.asarray(Image.fromarray(frame).resize((out_w, out_h), Image.BILINEAR))
        frames.append(frame)
    return Image.fromarray(np.concatenate(frames, axis=0))


def build_episode_controls(ep_info: dict, args):
    intrinsics = get_color_intrinsics()
    control_per_cam = {}
    ray_o_per_cam = {}
    ray_d_per_cam = {}
    intrinsic_by_cam = {}

    for camera in args.camera_names:
        k = scale_intrinsic(
            intrinsics[camera],
            args.intrinsic_source_height,
            args.intrinsic_source_width,
            args.view_height,
            args.view_width,
        )
        intrinsic_by_cam[camera] = k
        control_per_cam[camera] = generate_traj_map(
            ep_info["traj_actions_by_cam"][camera],
            ep_info["traj_w2c_by_cam"][camera],
            k,
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

        ray_o_seq = []
        ray_d_seq = []
        for t in range(ep_info["T_ds"]):
            c2w = np.linalg.inv(ep_info["ray_w2c_by_cam"][camera][t]).astype(np.float32)
            ray_o, ray_d = generate_raymap(k, c2w, args.view_height, args.view_width)
            ray_o_seq.append(to_uint8_img(ray_o, args.ray_o_vmin, args.ray_o_vmax))
            ray_d_seq.append(to_uint8_img(ray_d, args.ray_d_vmin, args.ray_d_vmax))
        ray_o_per_cam[camera] = ray_o_seq
        ray_d_per_cam[camera] = ray_d_seq

    control_video, ray_map_o, ray_map_d = [], [], []
    for t in range(ep_info["T_ds"]):
        control_video.append(Image.fromarray(np.concatenate([control_per_cam[c][t] for c in args.camera_names], axis=0)))
        ray_map_o.append(Image.fromarray(np.concatenate([ray_o_per_cam[c][t] for c in args.camera_names], axis=0)))
        ray_map_d.append(Image.fromarray(np.concatenate([ray_d_per_cam[c][t] for c in args.camera_names], axis=0)))
    return control_video, ray_map_o, ray_map_d, intrinsic_by_cam


def get_chunk_by_ids_with_pad(frames: Sequence[Image.Image], frame_ids: Sequence[int]) -> List[Image.Image]:
    return [frames[min(max(int(idx), 0), len(frames) - 1)] for idx in frame_ids]


def latent_frame_groups(num_frames: int, temporal_downsample: int) -> List[List[int]]:
    latent_t = (num_frames - 1) // temporal_downsample + 1
    groups = [[0] * temporal_downsample]
    for i in range(1, latent_t):
        start = 1 + (i - 1) * temporal_downsample
        groups.append([min(start + j, num_frames - 1) for j in range(temporal_downsample)])
    return groups


def raw_stack4_latent_intrinsic(image_intrinsic, image_height: int, image_width: int, spatial_downsample: int):
    if image_height % spatial_downsample != 0 or image_width % spatial_downsample != 0:
        raise ValueError(
            f"raw_stack4 requires view size divisible by {spatial_downsample}, "
            f"got view_height={image_height}, view_width={image_width}."
        )
    latent_h = image_height // spatial_downsample
    latent_w = image_width // spatial_downsample
    latent_intrinsic = image_intrinsic.copy()
    latent_intrinsic[0, 0] *= latent_w / float(image_width)
    latent_intrinsic[0, 2] *= latent_w / float(image_width)
    latent_intrinsic[1, 1] *= latent_h / float(image_height)
    latent_intrinsic[1, 2] *= latent_h / float(image_height)
    return latent_intrinsic, latent_h, latent_w


def build_raw_stack4_raymap_tensors(ep_info: dict, intrinsic_by_cam: dict, condition_ids: Sequence[int], args):
    groups = latent_frame_groups(len(condition_ids), args.vae_temporal_downsample)
    per_cam = {}
    for camera in args.camera_names:
        latent_intrinsic, latent_h, latent_w = raw_stack4_latent_intrinsic(
            intrinsic_by_cam[camera],
            args.view_height,
            args.view_width,
            args.vae_spatial_downsample,
        )
        per_cam[camera] = {
            "latent_intrinsic": latent_intrinsic,
            "latent_h": latent_h,
            "latent_w": latent_w,
        }

    ray_o_slots = []
    ray_d_slots = []
    for group in groups:
        ray_o_views = []
        ray_d_views = []
        for camera in args.camera_names:
            info = per_cam[camera]
            ray_o_group = []
            ray_d_group = []
            for pos in group:
                fid = min(max(int(condition_ids[pos]), 0), ep_info["T_ds"] - 1)
                c2w = np.linalg.inv(ep_info["ray_w2c_by_cam"][camera][fid]).astype(np.float32)
                ray_o, ray_d = generate_raymap(
                    info["latent_intrinsic"],
                    c2w,
                    info["latent_h"],
                    info["latent_w"],
                )
                ray_o_group.append(ray_o)
                ray_d_group.append(ray_d)
            ray_o_views.append(np.concatenate(ray_o_group, axis=-1))
            ray_d_views.append(np.concatenate(ray_d_group, axis=-1))
        ray_o_slots.append(np.concatenate(ray_o_views, axis=0))
        ray_d_slots.append(np.concatenate(ray_d_views, axis=0))

    ray_o = torch.from_numpy(np.stack(ray_o_slots, axis=0)).permute(3, 0, 1, 2).float()
    ray_d = torch.from_numpy(np.stack(ray_d_slots, axis=0)).permute(3, 0, 1, 2).float()
    return ray_o, ray_d


def save_video(frames: Sequence[Image.Image], path: Path, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))


def add_caption(frame: Image.Image, text: str, caption_h: int = 34) -> Image.Image:
    frame = frame.convert("RGB")
    canvas = Image.new("RGB", (frame.width, frame.height + caption_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), text, fill=(255, 255, 255))
    canvas.paste(frame, (0, caption_h))
    return canvas


def make_fourway_video(experiment_frames: Dict[str, List[Image.Image]], output_path: Path, fps: int):
    order = [
        ("old_steps5", "old / steps=5"),
        ("old_steps1", "old / steps=1"),
        ("new_steps5", "new / steps=5"),
        ("new_steps1", "new / steps=1"),
    ]
    min_len = min(len(experiment_frames[name]) for name, _ in order)
    out_frames = []
    for idx in range(min_len):
        cells = [add_caption(experiment_frames[name][idx], label) for name, label in order]
        max_h = max(cell.height for cell in cells)
        padded = []
        for cell in cells:
            if cell.height == max_h:
                padded.append(cell)
                continue
            pad = Image.new("RGB", (cell.width, max_h), (0, 0, 0))
            pad.paste(cell, (0, 0))
            padded.append(pad)
        canvas = Image.new("RGB", (sum(cell.width for cell in padded), max_h), (0, 0, 0))
        x = 0
        for cell in padded:
            canvas.paste(cell, (x, 0))
            x += cell.width
        out_frames.append(canvas)
    save_video(out_frames, output_path, fps=fps)


def run_experiment(
    pipe,
    experiment_name: str,
    raymap_mode: str,
    num_inference_steps: int,
    ep_info: dict,
    first_frame: Image.Image,
    control_video: Sequence[Image.Image],
    ray_map_o: Sequence[Image.Image],
    ray_map_d: Sequence[Image.Image],
    intrinsic_by_cam: dict,
    args,
):
    generated_frames = [first_frame]
    total_height = args.view_height * len(args.camera_names)
    with StageTimer(pipe, args.device) as timer:
        for chunk_idx in range(args.num_chunks):
            horizon_start_idx = 1 + chunk_idx * args.horizon_frames
            valid_count = args.horizon_frames
            if chunk_idx == 0:
                context_video = [first_frame] * args.context_frames
                condition_ids = [0] * args.context_frames + list(range(1, 1 + args.horizon_frames))
            else:
                context_video = [first_frame] + generated_frames[-(args.context_frames - 1):]
                condition_ids = [0]
                condition_ids += list(range(horizon_start_idx - (args.context_frames - 1), horizon_start_idx))
                condition_ids += list(range(horizon_start_idx, horizon_start_idx + args.horizon_frames))

            kwargs = {
                "prompt": args.prompt,
                "input_video": context_video,
                "vace_video": get_chunk_by_ids_with_pad(control_video, condition_ids),
                "height": total_height,
                "width": args.view_width,
                "num_frames": args.num_frames,
                "num_inference_steps": num_inference_steps,
                "cfg_scale": args.cfg_scale,
                "seed": args.seed + chunk_idx,
                "tiled": args.tiled,
                "progress_bar_cmd": lambda x: x,
            }
            if raymap_mode == "raw_stack4":
                raw_ray_o, raw_ray_d = build_raw_stack4_raymap_tensors(ep_info, intrinsic_by_cam, condition_ids, args)
                kwargs["ray_map_o"] = raw_ray_o
                kwargs["ray_map_d"] = raw_ray_d
            elif raymap_mode == "image":
                kwargs["ray_map_o"] = get_chunk_by_ids_with_pad(ray_map_o, condition_ids)
                kwargs["ray_map_d"] = get_chunk_by_ids_with_pad(ray_map_d, condition_ids)
            else:
                raise ValueError(f"Unsupported raymap_mode={raymap_mode}")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timer.start_chunk(experiment_name, chunk_idx)
            t0 = time.perf_counter()
            output = pipe(**kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            timer.end_chunk(elapsed, valid_count)

            new_frames = output[args.context_frames: args.context_frames + valid_count]
            generated_frames.extend([frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame)) for frame in new_frames])

            rec = timer.records[-1]
            print(
                f"[{experiment_name}] chunk {chunk_idx + 1}/{args.num_chunks} "
                f"total={rec['total_s']:.3f}s vae_enc={rec['vae_encode_s']:.3f}s "
                f"dit={rec['dit_s']:.3f}s vae_dec={rec['vae_decode_s']:.3f}s "
                f"other={rec['other_s']:.3f}s fps={rec['chunk_fps']:.3f}"
            )
    return generated_frames, timer.records


def write_timing_outputs(records: List[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "timings.json"
    csv_path = output_dir / "timings.csv"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "experiment",
        "chunk_idx",
        "valid_frames",
        "total_s",
        "vae_encode_s",
        "vae_encode_calls",
        "dit_s",
        "dit_calls",
        "vae_decode_s",
        "vae_decode_calls",
        "other_s",
        "chunk_fps",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({key: rec.get(key, "") for key in fields})

    print(f"[save] {json_path}")
    print(f"[save] {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/data/zsq/rollout_openpi_0601_6000_step3_2")
    parser.add_argument("--output_dir", type=str, default="/data/zsq/diffsynth-studio-wujie-ppu/outputs/arxx5_wm_speed_benchmark")
    parser.add_argument("--episode_index", type=int, default=None)
    parser.add_argument("--episode_pos", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=10)

    parser.add_argument("--prompt", type=str, default="机械臂按照要求移动夹爪执行任务")
    parser.add_argument("--camera_names", nargs="+", default=["head", "left_wrist", "right_wrist"])
    parser.add_argument("--camera_axis_mode", type=str, default="identity")
    parser.add_argument("--original_hz", type=int, default=30)
    parser.add_argument("--target_hz", type=int, default=30)
    parser.add_argument("--context_frames", type=int, default=5)
    parser.add_argument("--horizon_frames", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=13)

    parser.add_argument("--view_height", type=int, default=224)
    parser.add_argument("--view_width", type=int, default=224)
    parser.add_argument("--intrinsic_source_height", type=int, default=480)
    parser.add_argument("--intrinsic_source_width", type=int, default=640)
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

    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vae_spatial_downsample", type=int, default=16)
    parser.add_argument("--vae_temporal_downsample", type=int, default=4)

    parser.add_argument("--old_vace_model_path", type=str, default="/data/zsq/diffsynth-studio-wujie-ppu/outputs/0604arxx5/epoch-49.safetensors")
    parser.add_argument("--new_vace_model_path", type=str, required=True)
    parser.add_argument(
        "--base_model_paths",
        type=str,
        default=json.dumps(
            [
                [
                    "/data/zsq/Wan_model/diffusion_pytorch_model-00001-of-00003.safetensors",
                    "/data/zsq/Wan_model/diffusion_pytorch_model-00002-of-00003.safetensors",
                    "/data/zsq/Wan_model/diffusion_pytorch_model-00003-of-00003.safetensors",
                ],
                "/data/zsq/Wan_model/models_t5_umt5-xxl-enc-bf16.pth",
                "/data/zsq/Wan_model/Wan2.2_VAE.pth",
            ]
        ),
    )

    parser.add_argument("--head_left_calib", type=str, default="/data/zsq/outputs/calib_eye_to_hand_head_left/result_eye_to_hand.json")
    parser.add_argument("--head_right_calib", type=str, default="/data/zsq/outputs/calib_eye_to_hand_head_right/result_eye_to_hand.json")
    parser.add_argument("--left_calib", type=str, default="/data/zsq/outputs/calib_eye_in_hand_left/result_eye_in_hand.json")
    parser.add_argument("--right_calib", type=str, default="/data/zsq/outputs/calib_eye_in_hand_right/result_eye_in_hand.json")
    parser.add_argument("--fk_urdf_path", type=str, default="/data/zsq/x5.urdf")
    parser.add_argument("--fk_offset_path", type=str, default="/data/zsq/RLinf/rlinf/envs/world_model/arxx5_fk_offsets.json")
    parser.add_argument("--fk_state_unit", type=str, default="rad", choices=["rad", "deg"])
    args = parser.parse_args()

    args.base_model_paths = json.loads(args.base_model_paths)
    if args.num_frames != args.context_frames + args.horizon_frames:
        raise ValueError("--num_frames must equal --context_frames + --horizon_frames.")
    if args.context_frames != 5:
        raise ValueError("This benchmark is aligned to context_frames=5.")
    if args.original_hz % args.target_hz != 0:
        raise ValueError("--original_hz must be divisible by --target_hz.")
    args.downsample_step = args.original_hz // args.target_hz
    return args


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_fk(
        urdf_path=args.fk_urdf_path,
        offset_path=args.fk_offset_path,
        state_unit=args.fk_state_unit,
    )

    if not Path(args.old_vace_model_path).exists():
        raise FileNotFoundError(f"old_vace_model_path does not exist: {args.old_vace_model_path}")
    if not Path(args.new_vace_model_path).exists():
        raise FileNotFoundError(f"new_vace_model_path does not exist: {args.new_vace_model_path}")

    episodes = discover_episodes(Path(args.base_path), args.camera_names)
    ep = select_episode(episodes, args.episode_index, args.episode_pos)
    ep_info = build_episode_info(ep, args)
    first_frame = load_first_multiview_frame(ep_info, args.camera_names, args.view_height, args.view_width)
    control_video, ray_map_o, ray_map_d, intrinsic_by_cam = build_episode_controls(ep_info, args)

    print(
        f"[episode] index={ep_info['episode_index']} frames={ep_info['T_ds']} "
        f"base_path={args.base_path} output_dir={output_dir}"
    )

    experiment_frames: Dict[str, List[Image.Image]] = {}
    all_records: List[dict] = []
    structures = [
        {
            "name": "old",
            "path": args.old_vace_model_path,
            "vace_in_dim": 144,
            "raymap_mode": "image",
            "view_embedding_num_views": 0,
        },
        {
            "name": "new",
            "path": args.new_vace_model_path,
            "vace_in_dim": 72,
            "raymap_mode": "raw_stack4",
            "view_embedding_num_views": len(args.camera_names),
        },
    ]

    for structure in structures:
        pipe = load_pipe(
            args,
            structure["path"],
            structure["vace_in_dim"],
            structure["view_embedding_num_views"],
        )
        for steps in (5, 1):
            experiment_name = f"{structure['name']}_steps{steps}"
            frames, records = run_experiment(
                pipe=pipe,
                experiment_name=experiment_name,
                raymap_mode=structure["raymap_mode"],
                num_inference_steps=steps,
                ep_info=ep_info,
                first_frame=first_frame,
                control_video=control_video,
                ray_map_o=ray_map_o,
                ray_map_d=ray_map_d,
                intrinsic_by_cam=intrinsic_by_cam,
                args=args,
            )
            experiment_frames[experiment_name] = frames
            all_records.extend(records)
            save_video(frames, output_dir / f"{experiment_name}.mp4", fps=args.fps)
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    make_fourway_video(experiment_frames, output_dir / "old_new_steps5_steps1_fourway.mp4", fps=args.fps)
    write_timing_outputs(all_records, output_dir)

    for name in ("old_steps5", "old_steps1", "new_steps5", "new_steps1"):
        rows = [rec for rec in all_records if rec["experiment"] == name]
        total = sum(rec["total_s"] for rec in rows)
        vae_encode = sum(rec["vae_encode_s"] for rec in rows)
        dit = sum(rec["dit_s"] for rec in rows)
        vae_decode = sum(rec["vae_decode_s"] for rec in rows)
        frames = sum(rec["valid_frames"] for rec in rows)
        print(
            f"[summary] {name}: total={total:.3f}s frames={frames} fps={frames / max(total, 1e-8):.3f} "
            f"vae_encode={vae_encode:.3f}s dit={dit:.3f}s vae_decode={vae_decode:.3f}s"
        )


if __name__ == "__main__":
    main()
