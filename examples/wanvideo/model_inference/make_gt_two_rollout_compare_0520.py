#!/usr/bin/env python3
import argparse
from itertools import chain
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def list_episode_keys(*rollout_dirs):
    common = None
    for root in rollout_dirs:
        keys = {p.name for p in Path(root).iterdir() if (p / "video.mp4").is_file()}
        common = keys if common is None else common & keys
    return sorted(common or [])


def open_reader(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    return imageio.get_reader(str(path), "ffmpeg")


def reader_len(reader):
    try:
        n = reader.count_frames()
        if n and n > 0:
            return int(n)
    except Exception:
        pass
    try:
        n = reader.get_length()
        if n != float("inf") and n > 0:
            return int(n)
    except Exception:
        pass
    return None


def np_to_pil(frame):
    return Image.fromarray(np.asarray(frame)).convert("RGB")


def fit_frame(img, size):
    if img.size == size:
        return img
    return img.resize(size, Image.BILINEAR)


def add_caption(img, caption, caption_h=36):
    w, h = img.size
    out = Image.new("RGB", (w, h + caption_h), (20, 20, 20))
    out.paste(img, (0, caption_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), caption, font=font)
    x = max(0, (w - (bbox[2] - bbox[0])) // 2)
    y = max(0, (caption_h - (bbox[3] - bbox[1])) // 2 - 1)
    draw.text((x, y), caption, fill=(255, 255, 255), font=font)
    return out


def concat_row(frames, captions):
    capped = [add_caption(frame, cap) for frame, cap in zip(frames, captions)]
    w = sum(frame.size[0] for frame in capped)
    h = max(frame.size[1] for frame in capped)
    out = Image.new("RGB", (w, h), (0, 0, 0))
    x = 0
    for frame in capped:
        out.paste(frame, (x, 0))
        x += frame.size[0]
    return out


def gt_video_paths(gt_val_root, episode_key, camera_names):
    task_id, episode_id = episode_key.split("-", 1)
    video_dir = Path(gt_val_root) / "observations" / task_id / episode_id / "videos"
    paths = [video_dir / f"{cam}_color.mp4" for cam in camera_names]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing GT videos for {episode_key}: {missing}")
    return paths


def make_gt_frame(frames, view_size):
    views = [fit_frame(np_to_pil(frame), view_size) for frame in frames]
    w, h = view_size
    canvas = Image.new("RGB", (w, h * len(views)), (0, 0, 0))
    for i, view in enumerate(views):
        canvas.paste(view, (0, i * h))
    return canvas


def downsampled_gt_iter(gt_readers, step):
    for raw_idx, frames in enumerate(zip(*(iter(reader) for reader in gt_readers))):
        if raw_idx % step == 0:
            yield frames


def make_compare_for_episode(args, episode_key):
    rollout_a_path = Path(args.rollout_a_dir) / episode_key / "video.mp4"
    rollout_b_path = Path(args.rollout_b_dir) / episode_key / "video.mp4"
    rollout_a_reader = open_reader(rollout_a_path)
    rollout_b_reader = open_reader(rollout_b_path)
    gt_readers = [open_reader(p) for p in gt_video_paths(args.gt_val_root, episode_key, args.camera_names)]

    try:
        rollout_a_iter = iter(rollout_a_reader)
        rollout_b_iter = iter(rollout_b_reader)
        first_a = next(rollout_a_iter, None)
        first_b = next(rollout_b_iter, None)
        if first_a is None or first_b is None:
            raise RuntimeError(f"Cannot read first rollout frame for {episode_key}")

        rollout_size = (int(first_a.shape[1]), int(first_a.shape[0]))
        view_w = args.view_width or rollout_size[0]
        view_h = args.view_height or (rollout_size[1] // len(args.camera_names))
        step = max(1, int(round(float(args.original_hz) / float(args.target_hz))))

        total_candidates = [reader_len(rollout_a_reader), reader_len(rollout_b_reader)]
        gt_raw_lens = [reader_len(reader) for reader in gt_readers]
        if all(n is not None for n in total_candidates + gt_raw_lens):
            total = min(total_candidates + [min(gt_raw_lens) // step])
            if args.max_frames is not None:
                total = min(total, args.max_frames)
        else:
            total = args.max_frames

        out_path = Path(args.output_dir) / f"{episode_key}_GT_{args.rollout_a_tag}_{args.rollout_b_tag}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        total_text = "unknown" if total is None else str(total)
        print(f"[START] {episode_key}: total_frames={total_text} -> {out_path}", flush=True)

        rollout_a_frames = chain([first_a], rollout_a_iter)
        rollout_b_frames = chain([first_b], rollout_b_iter)
        gt_frames = downsampled_gt_iter(gt_readers, step)
        count = 0
        with imageio.get_writer(str(out_path), fps=args.fps, codec="libx264", quality=args.quality) as writer:
            for gt_tuple, a_frame, b_frame in zip(gt_frames, rollout_a_frames, rollout_b_frames):
                if total is not None and count >= total:
                    break
                gt_pil = fit_frame(make_gt_frame(gt_tuple, (view_w, view_h)), rollout_size)
                a_pil = fit_frame(np_to_pil(a_frame), rollout_size)
                b_pil = fit_frame(np_to_pil(b_frame), rollout_size)
                merged = concat_row(
                    [gt_pil, a_pil, b_pil],
                    ["GT", args.rollout_a_caption, args.rollout_b_caption],
                )
                writer.append_data(np.asarray(merged))
                count += 1
                if count % args.progress_interval == 0:
                    print(f"[PROGRESS] {episode_key}: {count}/{total_text}", flush=True)

        print(f"[DONE] {episode_key}: frames={count} -> {out_path}", flush=True)
    finally:
        rollout_a_reader.close()
        rollout_b_reader.close()
        for reader in gt_readers:
            reader.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_val_root", default="/mnt/workspace/zsq/Agi2024subset_split/val")
    parser.add_argument(
        "--rollout_a_dir",
        default="/mnt/workspace/zsq/Agi2024subset_split/val_wan22_vace_ti2v2_stage2_qian1t=0_0520_perspective_raymap",
    )
    parser.add_argument(
        "--rollout_b_dir",
        default="/mnt/workspace/zsq/Agi2024subset_split/val_wan22_vace_ti2v2_stage2_qian2t=0_0520_perspective_raymap",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/workspace/zsq/Agi2024subset_split/compare_GT_qian1t0_qian2t0_0520",
    )
    parser.add_argument("--camera_names", nargs="+", default=["head", "hand_left", "hand_right"])
    parser.add_argument("--rollout_a_caption", default="WAN qian1t=0 0520")
    parser.add_argument("--rollout_b_caption", default="WAN qian2t=0 0520")
    parser.add_argument("--rollout_a_tag", default="qian1t0")
    parser.add_argument("--rollout_b_tag", default="qian2t0")
    parser.add_argument("--original_hz", type=float, default=30)
    parser.add_argument("--target_hz", type=float, default=5)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=7)
    parser.add_argument("--view_width", type=int, default=None)
    parser.add_argument("--view_height", type=int, default=None)
    parser.add_argument("--episode_keys", nargs="+", default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--progress_interval", type=int, default=50)
    args = parser.parse_args()

    episode_keys = args.episode_keys or list_episode_keys(args.rollout_a_dir, args.rollout_b_dir)
    if not episode_keys:
        raise RuntimeError("No common episodes found in rollout directories.")
    print(f"[INFO] episodes={episode_keys}", flush=True)
    print(f"[INFO] output_dir={args.output_dir}", flush=True)
    for episode_key in episode_keys:
        make_compare_for_episode(args, episode_key)


if __name__ == "__main__":
    main()
