#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_CAMERAS = ("head", "hand_left", "hand_right")


def run_cmd(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def resolve_binary(name, override=None):
    if override:
        return override
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"Cannot find executable: {name}")
    return path


def probe_codec(video_path, ffprobe_bin):
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "json",
        str(video_path),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}:\n{result.stderr.strip()}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    return streams[0].get("codec_name", "")


def transcode_to_h264(src, dst, ffmpeg_bin, crf, preset, overwrite):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y" if overwrite else "-n",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(dst),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src} -> {dst}:\n{result.stderr.strip()}")


def copy_or_link_existing_h264(src, dst, mode, overwrite):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            return "skip_existing"
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported h264_existing_mode: {mode}")
    return mode


def collect_camera_videos(input_root, splits, cameras, limit=None):
    wanted = {f"{cam}_color.mp4" for cam in cameras}
    videos = []
    for split in splits:
        split_root = input_root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Missing split directory: {split_root}")
        for video_dir in split_root.rglob("videos"):
            if not video_dir.is_dir():
                continue
            for name in sorted(wanted):
                path = video_dir / name
                if path.exists():
                    videos.append(path)
                    if limit is not None and len(videos) >= limit:
                        return videos
    return videos


def process_one(src, input_root, output_root, ffmpeg_bin, ffprobe_bin, args):
    rel = src.relative_to(input_root)
    dst = output_root / rel
    if dst.exists() and not args.overwrite:
        return ("skip_existing", src, dst, "")

    codec = "unknown"
    if args.dry_run and not args.probe_in_dry_run:
        return ("dry_transcode_or_link", src, dst, codec)

    if not args.skip_probe:
        codec = probe_codec(src, ffprobe_bin).lower()
        if codec == "h264" and not args.force_transcode:
            if args.dry_run:
                return ("dry_existing_h264", src, dst, codec)
            action = copy_or_link_existing_h264(src, dst, args.h264_existing_mode, args.overwrite)
            return (action, src, dst, codec)

    if args.dry_run:
        return ("dry_transcode", src, dst, codec)
    transcode_to_h264(src, dst, ffmpeg_bin, args.crf, args.preset, args.overwrite)
    return ("transcoded", src, dst, codec)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create an H.264 video-only copy of Agi2024subset_split for the three "
            "Wan multiview cameras: head, hand_left, hand_right."
        )
    )
    parser.add_argument(
        "--input_root",
        default="/mnt/workspace/zsq/Agi2024subset_split",
        help="Original dataset root containing train/ and val/.",
    )
    parser.add_argument(
        "--output_root",
        default="/mnt/workspace/zsq/Agi2024subset_split_h264",
        help="Output H.264 dataset root.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to process.")
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS), help="Camera names to keep.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel ffmpeg jobs.")
    parser.add_argument("--crf", type=int, default=18, help="libx264 CRF.")
    parser.add_argument("--preset", default="medium", help="libx264 preset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned work without writing files.")
    parser.add_argument("--probe_in_dry_run", action="store_true", help="Run ffprobe even in dry-run mode.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N selected videos.")
    parser.add_argument("--skip_probe", action="store_true", help="Skip ffprobe and transcode every selected video.")
    parser.add_argument("--force_transcode", action="store_true", help="Transcode even if source is already h264.")
    parser.add_argument(
        "--h264_existing_mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to place already-H.264 source videos in the output tree.",
    )
    parser.add_argument("--ffmpeg_bin", default=None, help="Optional ffmpeg path.")
    parser.add_argument("--ffprobe_bin", default=None, help="Optional ffprobe path.")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    ffmpeg_bin = resolve_binary("ffmpeg", args.ffmpeg_bin)
    ffprobe_bin = None if args.skip_probe else resolve_binary("ffprobe", args.ffprobe_bin)

    videos = collect_camera_videos(input_root, args.splits, args.cameras, limit=args.limit)
    print(f"[INFO] input_root: {input_root}")
    print(f"[INFO] output_root: {output_root}")
    print(f"[INFO] splits: {args.splits}")
    print(f"[INFO] cameras: {args.cameras}")
    print(f"[INFO] selected videos: {len(videos)}")
    if not videos:
        raise RuntimeError("No selected camera videos found.")

    counts = {}
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = [
            executor.submit(process_one, src, input_root, output_root, ffmpeg_bin, ffprobe_bin, args)
            for src in videos
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            status, src, dst, codec = future.result()
            counts[status] = counts.get(status, 0) + 1
            print(f"[{idx}/{len(futures)}] {status} codec={codec}: {src} -> {dst}")

    print("[INFO] summary:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    print("[INFO] done")


if __name__ == "__main__":
    main()
