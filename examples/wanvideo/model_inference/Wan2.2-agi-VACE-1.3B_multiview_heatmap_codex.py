#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

WUJIE_REPO = "/data/zsq/diffsynth-studio-wujie-ppu"
THIS_DIR = Path(__file__).resolve().parent
FPS_SCRIPT = THIS_DIR / "Wan2.2-agi-VACE-1.3B_multiview_fps_codex.py"

if WUJIE_REPO not in sys.path:
    sys.path.insert(0, WUJIE_REPO)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from diffsynth import save_video


def load_fps_module():
    spec = importlib.util.spec_from_file_location("wan_multiview_fps_codex", str(FPS_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper script: {FPS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fps_mod = load_fps_module()


def parse_int_list(value):
    if value is None or value.strip() == "":
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def token_frame_index(frame_idx, latent_t):
    if latent_t <= 1:
        return 0
    if frame_idx == 0:
        return 0
    return min(1 + (frame_idx - 1) // 4, latent_t - 1)


def to_uint8_frame(image, width, height):
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB").resize((width, height), Image.BILINEAR))
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((width, height), Image.BILINEAR))
    return arr.astype(np.uint8)


def heat_to_color_image(heat_2d, width, height):
    heat_2d = np.asarray(heat_2d, dtype=np.float32)
    heat_2d = np.nan_to_num(heat_2d, nan=0.0, posinf=1.0, neginf=0.0)
    gray = Image.fromarray(np.clip(heat_2d * 255.0, 0, 255).astype(np.uint8), mode="L")
    gray = gray.resize((width, height), Image.BILINEAR)
    return ImageOps.colorize(gray, black="#101020", mid="#f4c430", white="#ff2f2f")


class VaceHintHeatmapCollector:
    def __init__(
        self,
        pipe,
        total_height,
        width,
        num_frames,
        selected_hint_indices,
        every_n_calls,
        max_records,
        alpha,
        fps,
    ):
        if not hasattr(pipe, "vace") or pipe.vace is None:
            raise RuntimeError("The loaded pipeline has no pipe.vace module.")
        self.pipe = pipe
        self.vace = pipe.vace
        self.total_height = int(total_height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.token_h = max(1, self.total_height // 16)
        self.token_w = max(1, self.width // 16)
        self.spatial_tokens = self.token_h * self.token_w
        self.latent_t = (self.num_frames - 1) // 4 + 1
        self.selected_hint_indices = list(selected_hint_indices)
        self.every_n_calls = max(1, int(every_n_calls))
        self.max_records = int(max_records)
        self.alpha = float(alpha)
        self.fps = int(fps)
        self.vace_layers = list(getattr(self.vace, "vace_layers", []))
        self.chunk_idx = 0
        self.call_idx = 0
        self.records = []
        self.control_frames = []
        self.original_forward = self.vace.forward
        self._installed = False

    def install(self):
        if self._installed:
            return

        def wrapped_forward(*args, **kwargs):
            hints = self.original_forward(*args, **kwargs)
            self.capture(hints)
            return hints

        self.vace.forward = wrapped_forward
        self._installed = True

    def uninstall(self):
        if self._installed:
            self.vace.forward = self.original_forward
            self._installed = False

    def start_chunk(self, chunk_idx, control_frames):
        self.chunk_idx = int(chunk_idx)
        self.call_idx = 0
        self.records = []
        self.control_frames = list(control_frames)

    def _selected_indices(self, hints):
        total = len(hints)
        selected = self.selected_hint_indices or [0, total // 2, total - 1]
        out = []
        for idx in selected:
            idx = total + idx if idx < 0 else idx
            if 0 <= idx < total and idx not in out:
                out.append(idx)
        return out

    def capture(self, hints):
        if not isinstance(hints, (tuple, list)):
            return
        call_idx = self.call_idx
        self.call_idx += 1
        if call_idx % self.every_n_calls != 0:
            return
        if self.max_records > 0 and len(self.records) >= self.max_records:
            return

        for hint_idx in self._selected_indices(hints):
            hint = hints[hint_idx]
            if not torch.is_tensor(hint):
                continue
            with torch.no_grad():
                heat = hint.detach().float().norm(dim=-1)
                if heat.ndim == 2:
                    heat = heat.mean(dim=0)
                heat = heat.flatten()
                usable = (heat.numel() // self.spatial_tokens) * self.spatial_tokens
                if usable <= 0:
                    continue
                heat = heat[-usable:].reshape(-1, self.token_h, self.token_w)
                if heat.shape[0] > self.latent_t:
                    heat = heat[-self.latent_t:]
                lo = heat.amin()
                hi = heat.amax()
                heat = (heat - lo) / (hi - lo + 1e-6)
                heat_np = heat.cpu().numpy().astype(np.float16)

            block_id = (
                int(self.vace_layers[hint_idx])
                if 0 <= hint_idx < len(self.vace_layers)
                else int(hint_idx)
            )
            self.records.append(
                {
                    "chunk_idx": self.chunk_idx,
                    "call_idx": call_idx,
                    "hint_idx": int(hint_idx),
                    "block_id": block_id,
                    "heat": heat_np,
                    "shape": list(hint.shape),
                }
            )

    def save_chunk(self, chunk_dir):
        chunk_dir = Path(chunk_dir)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "chunk_idx": self.chunk_idx,
            "num_vace_calls": self.call_idx,
            "total_height": self.total_height,
            "width": self.width,
            "num_frames": self.num_frames,
            "token_h": self.token_h,
            "token_w": self.token_w,
            "latent_t": self.latent_t,
            "vace_layers": self.vace_layers,
            "records": [],
        }

        blank = np.zeros((self.total_height, self.width, 3), dtype=np.uint8)
        for rec in self.records:
            heat = rec["heat"].astype(np.float32)
            heat_frames = []
            overlay_frames = []
            for frame_idx in range(self.num_frames):
                latent_idx = token_frame_index(frame_idx, heat.shape[0])
                color = heat_to_color_image(heat[latent_idx], self.width, self.total_height)
                color_arr = np.asarray(color.convert("RGB"), dtype=np.float32)
                if frame_idx < len(self.control_frames):
                    base = to_uint8_frame(self.control_frames[frame_idx], self.width, self.total_height)
                else:
                    base = blank
                overlay = np.clip((1.0 - self.alpha) * base.astype(np.float32) + self.alpha * color_arr, 0, 255)
                heat_frames.append(color)
                overlay_frames.append(Image.fromarray(overlay.astype(np.uint8)))

            name = f"step_{rec['call_idx']:03d}_hint_{rec['hint_idx']:02d}_block_{rec['block_id']:02d}"
            heat_path = chunk_dir / f"{name}_heat_codex.mp4"
            overlay_path = chunk_dir / f"{name}_overlay_codex.mp4"
            png_path = chunk_dir / f"{name}_overlay_frame0_codex.png"
            save_video(heat_frames, str(heat_path), fps=self.fps, quality=5)
            save_video(overlay_frames, str(overlay_path), fps=self.fps, quality=5)
            overlay_frames[0].save(png_path)
            metadata["records"].append(
                {
                    "call_idx": rec["call_idx"],
                    "hint_idx": rec["hint_idx"],
                    "block_id": rec["block_id"],
                    "source_hint_shape": rec["shape"],
                    "heat_shape": list(heat.shape),
                    "heat_video": str(heat_path),
                    "overlay_video": str(overlay_path),
                    "overlay_frame0": str(png_path),
                }
            )

        meta_path = chunk_dir / "vace_hint_heatmap_metadata_codex.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return metadata


def run_episode_heatmap(pipe, ep_info, args):
    first_frame = fps_mod.load_first_multiview_frame(
        ep_info, args.camera_names, args.view_height, args.view_width
    )
    control_video, ray_map_o, ray_map_d = fps_mod.build_episode_controls(ep_info, args)

    total_frames = ep_info["T_ds"]
    num_chunks_all = max(1, int(math.ceil(max(total_frames - 1, 0) / float(args.predict_frames))))
    num_chunks = min(args.num_chunks, num_chunks_all)
    total_height = args.view_height * len(args.camera_names)
    ep_out_dir = Path(args.output_dir) / ep_info["episode_key"]
    ep_out_dir.mkdir(parents=True, exist_ok=True)

    collector = VaceHintHeatmapCollector(
        pipe=pipe,
        total_height=total_height,
        width=args.view_width,
        num_frames=args.num_frames,
        selected_hint_indices=parse_int_list(args.heatmap_hint_indices),
        every_n_calls=args.heatmap_every_n_calls,
        max_records=args.heatmap_max_records_per_chunk,
        alpha=args.heatmap_alpha,
        fps=args.fps,
    )
    collector.install()

    current_context = first_frame
    summary = {
        "episode_key": ep_info["episode_key"],
        "vace_model_path": args.vace_model_path,
        "output_dir": str(ep_out_dir),
        "num_chunks_run": 0,
        "chunks": [],
    }

    try:
        for chunk_idx in range(num_chunks):
            context_idx = chunk_idx * args.predict_frames
            remaining_predictions = max(total_frames - 1 - context_idx, 0)
            valid_prediction_count = min(args.predict_frames, remaining_predictions)
            if valid_prediction_count <= 0:
                break

            vace_video = fps_mod.get_chunk_with_pad(control_video, context_idx, args.num_frames)
            collector.start_chunk(chunk_idx, vace_video)
            pipe_kwargs = {
                "prompt": args.prompt,
                "vace_video": vace_video,
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
                    fps_mod.get_latent_chunk_with_pad(ray_map_o, context_idx, args.num_frames)
                    if args.raymap_mode == "latent"
                    else fps_mod.get_chunk_with_pad(ray_map_o, context_idx, args.num_frames)
                )
                pipe_kwargs["ray_map_d"] = (
                    fps_mod.get_latent_chunk_with_pad(ray_map_d, context_idx, args.num_frames)
                    if args.raymap_mode == "latent"
                    else fps_mod.get_chunk_with_pad(ray_map_d, context_idx, args.num_frames)
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            generated = pipe(**pipe_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            chunk_dir = ep_out_dir / f"chunk_{chunk_idx:03d}"
            chunk_meta = collector.save_chunk(chunk_dir)
            if args.save_control_video:
                save_video(vace_video, str(chunk_dir / "vace_control_video_codex.mp4"), fps=args.fps, quality=5)
            if args.save_generated_video:
                save_video(generated, str(chunk_dir / "generated_video_codex.mp4"), fps=args.fps, quality=5)

            new_frames = generated[1: 1 + valid_prediction_count]
            current_context = new_frames[-1] if len(new_frames) > 0 else current_context
            summary["num_chunks_run"] += 1
            summary["chunks"].append(
                {
                    "chunk_idx": chunk_idx,
                    "context_idx": context_idx,
                    "valid_prediction_count": valid_prediction_count,
                    "seconds": elapsed,
                    "num_vace_calls": chunk_meta["num_vace_calls"],
                    "num_heatmap_records": len(chunk_meta["records"]),
                    "chunk_dir": str(chunk_dir),
                }
            )
            print(
                f"[HEATMAP] episode={ep_info['episode_key']} chunk={chunk_idx + 1}/{num_chunks} "
                f"time={elapsed:.3f}s vace_calls={chunk_meta['num_vace_calls']} "
                f"records={len(chunk_meta['records'])} out={chunk_dir}",
                flush=True,
            )
    finally:
        collector.uninstall()

    summary_path = ep_out_dir / "vace_hint_heatmap_summary_codex.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[HEATMAP] wrote {summary_path}", flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", type=str, default="/data/zsq/agibot2024/Agi2024subset_split/val")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)
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
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--traj_radius", type=int, default=3)
    parser.add_argument("--traj_min_radius", type=float, default=2.0)
    parser.add_argument("--traj_max_radius", type=float, default=7.0)
    parser.add_argument("--traj_ref_depth", type=float, default=1.0)
    parser.add_argument("--traj_near_depth", type=float, default=0.3)
    parser.add_argument("--traj_far_depth", type=float, default=2.0)
    parser.add_argument("--output_raymap", action="store_true", default=False)
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/zsq/diffsynth-studio-wujie-ppu/outputs/Wan2.1-VACE-1.3B-multiview-constant-noraymap/vace_hint_heatmap_codex",
    )
    parser.add_argument(
        "--heatmap_hint_indices",
        type=str,
        default="0,4,8,12,14",
        help="Comma-separated VACE hint indices. Use -1 for the last hint.",
    )
    parser.add_argument("--heatmap_every_n_calls", type=int, default=1)
    parser.add_argument("--heatmap_max_records_per_chunk", type=int, default=128)
    parser.add_argument("--heatmap_alpha", type=float, default=0.45)
    parser.add_argument("--save_control_video", action="store_true", default=True)
    parser.add_argument("--no_save_control_video", dest="save_control_video", action="store_false")
    parser.add_argument("--save_generated_video", action="store_true", default=True)
    parser.add_argument("--no_save_generated_video", dest="save_generated_video", action="store_false")
    args = parser.parse_args()
    if args.original_hz % args.target_hz != 0:
        raise ValueError("--original_hz must be divisible by --target_hz.")
    if args.num_frames != args.predict_frames + 1:
        raise ValueError("--num_frames must equal --predict_frames + 1.")
    args.downsample_step = args.original_hz // args.target_hz
    return args


def main():
    args = parse_args()
    episodes = fps_mod.discover_episodes(args.val_path)
    if not (0 <= args.episode_index < len(episodes)):
        raise IndexError(f"--episode_index out of range: {args.episode_index}, total episodes={len(episodes)}")
    ep = episodes[args.episode_index]
    ep_info = fps_mod.build_episode_info(ep, args.downsample_step, args.camera_names)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    pipe = fps_mod.load_pipe(args.base_model_paths, args.vace_model_path, args.device)
    run_episode_heatmap(pipe, ep_info, args)


if __name__ == "__main__":
    main()
