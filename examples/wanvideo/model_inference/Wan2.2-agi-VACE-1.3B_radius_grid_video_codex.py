import argparse
import os

import imageio
import imageio.v3 as iio
import numpy as np
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont


def read_video_frames(video_path):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()
        del vr
        if frames.shape[-1] == 4:
            frames = frames[..., :3]
        return [frame for frame in frames]
    except Exception:
        reader = imageio.get_reader(video_path, format="ffmpeg")
        frames = []
        try:
            for frame in reader:
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]
                frames.append(frame)
        finally:
            reader.close()
        return frames


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


def load_gt_multiview_frames(video_dir, camera_names, downsample_step, target_size):
    view_width, view_height = target_size
    per_cam_frames = []
    min_frames = None
    for cam in camera_names:
        video_path = os.path.join(video_dir, f"{cam}_color.mp4")
        frames = read_video_frames(video_path)
        if not frames:
            raise RuntimeError(f"No frames found in {video_path}")
        if min_frames is None:
            min_frames = len(frames)
        else:
            min_frames = min(min_frames, len(frames))
        per_cam_frames.append(frames)

    gt_frames = []
    for idx in range(0, min_frames, downsample_step):
        parts = []
        for frames in per_cam_frames:
            frame = Image.fromarray(frames[idx]).resize((view_width, view_height), Image.BILINEAR)
            parts.append(np.array(frame))
        gt_frames.append(np.concatenate(parts, axis=0))
    return gt_frames


def resize_frames(frames, target_hw):
    target_h, target_w = target_hw
    resized = []
    for frame in frames:
        if frame.shape[0] == target_h and frame.shape[1] == target_w:
            resized.append(frame)
            continue
        pil = Image.fromarray(frame).resize((target_w, target_h), Image.BILINEAR)
        resized.append(np.array(pil))
    return resized


def build_labeled_column(frame, label, label_height, font):
    frame_h, frame_w = frame.shape[:2]
    canvas = Image.new("RGB", (frame_w, frame_h + label_height), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = max((frame_w - text_w) // 2, 0)
    text_y = max((label_height - text_h) // 2 - 2, 0)
    draw.text((text_x, text_y), label, fill=(240, 240, 240), font=font)
    canvas.paste(Image.fromarray(frame), (0, label_height))
    return np.array(canvas)


def load_font(font_size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, font_size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_key", type=str, default="357-712726")
    parser.add_argument("--gt_video_dir", type=str, default="/mnt/workspace/zsq/Agi2024subset_split/val/observations/357/712726/videos")
    parser.add_argument("--constant_video", type=str, default="/mnt/workspace/zsq/Agi2024subset_split/val_gen_constant_noraymap/357-712726/video.mp4")
    parser.add_argument("--depth_norm_video", type=str, default="/mnt/workspace/zsq/Agi2024subset_split/val_gen_depth_norm_noraymap/357-712726/video.mp4")
    parser.add_argument("--perspective_video", type=str, default="/mnt/workspace/zsq/Agi2024subset_split/val_gen_perspective_noraymap/357-712726/video.mp4")
    parser.add_argument("--perspective_raymap_video", type=str, default="/mnt/workspace/zsq/Agi2024subset_split/val_gen_perspective_raymap/357-712726/video.mp4")
    parser.add_argument("--output_path", type=str, default="/mnt/workspace/zsq/Agi2024subset_split/radius_mode_compare_357-712726.mp4")
    parser.add_argument("--camera_names", nargs="+", default=["head", "hand_left", "hand_right"])
    parser.add_argument("--original_hz", type=int, default=30)
    parser.add_argument("--target_hz", type=int, default=5)
    parser.add_argument("--view_width", type=int, default=512)
    parser.add_argument("--view_height", type=int, default=320)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--label_height", type=int, default=54)
    args = parser.parse_args()

    if args.original_hz % args.target_hz != 0:
        raise ValueError("--original_hz must be divisible by --target_hz")

    downsample_step = args.original_hz // args.target_hz
    target_size = (args.view_width, args.view_height)

    gt_frames = load_gt_multiview_frames(
        video_dir=args.gt_video_dir,
        camera_names=args.camera_names,
        downsample_step=downsample_step,
        target_size=target_size,
    )

    video_specs = [
        ("GT", gt_frames),
        ("constant", read_video_frames(args.constant_video)),
        ("depth-norm", read_video_frames(args.depth_norm_video)),
        ("perspective", read_video_frames(args.perspective_video)),
        ("perspective-with-raymap", read_video_frames(args.perspective_raymap_video)),
    ]

    if not video_specs[1][1]:
        raise RuntimeError(f"No frames found in {args.constant_video}")
    ref_h, ref_w = video_specs[1][1][0].shape[:2]
    normalized_specs = []
    for label, frames in video_specs:
        if not frames:
            raise RuntimeError(f"No frames available for {label}")
        normalized_specs.append((label, resize_frames(frames, (ref_h, ref_w))))

    total_frames = min(len(frames) for _, frames in normalized_specs)
    if total_frames == 0:
        raise RuntimeError("No overlapping frames to write")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    font = load_font(font_size=28)
    writer = imageio.get_writer(args.output_path, fps=args.fps)
    try:
        for frame_idx in range(total_frames):
            columns = []
            for label, frames in normalized_specs:
                columns.append(build_labeled_column(frames[frame_idx], label, args.label_height, font))
            writer.append_data(np.concatenate(columns, axis=1))
    finally:
        writer.close()

    print(f"[DONE] Saved {total_frames} frames to {args.output_path}")


if __name__ == "__main__":
    main()
