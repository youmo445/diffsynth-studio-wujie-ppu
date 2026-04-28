import argparse
import json
import shutil
import subprocess
from pathlib import Path


def run_cmd(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def resolve_binary(name: str, override: str | None):
    if override:
        return override
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"Cannot find executable: {name}")
    return path


def probe_codec(video_path: Path, ffprobe_bin: str):
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


def transcode_to_h264(src: Path, dst: Path, crf: int, preset: str, ffmpeg_bin: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src),
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


def copy_non_video_files(input_root: Path, output_root: Path, pattern: str, overwrite: bool):
    copied = 0
    skipped = 0
    for src in sorted(input_root.rglob("*")):
        if not src.is_file():
            continue
        if src.match(pattern):
            continue
        rel = src.relative_to(input_root)
        dst = output_root / rel
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Batch transcode AgiBot mp4 videos to H.264 for decord VideoReader compatibility."
    )
    parser.add_argument("--input_root", required=True, help="Input dataset root, e.g. /mnt/.../Agi2024subset_split/val")
    parser.add_argument("--output_root", required=True, help="Output dataset root for transcoded videos")
    parser.add_argument("--pattern", default="*_color.mp4", help="Video filename glob pattern")
    parser.add_argument("--only_codec", default="av1", help="Only transcode files whose codec matches this value")
    parser.add_argument("--crf", type=int, default=18, help="libx264 CRF value")
    parser.add_argument("--preset", default="medium", help="libx264 preset")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output videos")
    parser.add_argument("--dry_run", action="store_true", help="Only print planned actions")
    parser.add_argument("--skip_probe", action="store_true", help="Skip ffprobe codec detection and transcode all matched files")
    parser.add_argument("--ffprobe_bin", default=None, help="Path to ffprobe executable")
    parser.add_argument("--ffmpeg_bin", default=None, help="Path to ffmpeg executable")
    parser.add_argument("--copy_non_video", action="store_true", help="Also copy non-video files to mirror the dataset structure")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    ffmpeg_bin = resolve_binary("ffmpeg", args.ffmpeg_bin)

    ffprobe_bin = None
    if not args.skip_probe:
        try:
            ffprobe_bin = resolve_binary("ffprobe", args.ffprobe_bin)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{e}. You can either provide --ffprobe_bin /path/to/ffprobe "
                f"or rerun with --skip_probe to transcode all matched videos."
            )

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    video_paths = sorted(input_root.rglob(args.pattern))
    print(f"[INFO] found {len(video_paths)} candidate videos under {input_root}")

    planned = []
    skipped_codec = 0
    skipped_existing = 0

    for src in video_paths:
        rel = src.relative_to(input_root)
        dst = output_root / rel
        codec = "unknown"
        if not args.skip_probe:
            codec = probe_codec(src, ffprobe_bin)

        if not args.skip_probe and args.only_codec and codec.lower() != args.only_codec.lower():
            skipped_codec += 1
            continue

        if dst.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        planned.append((src, dst, codec))

    print(f"[INFO] codec-mismatch skipped: {skipped_codec}")
    print(f"[INFO] existing-output skipped: {skipped_existing}")
    print(f"[INFO] will transcode: {len(planned)}")

    for idx, (src, dst, codec) in enumerate(planned, start=1):
        print(f"[{idx}/{len(planned)}] {codec}: {src} -> {dst}")
        if not args.dry_run:
            transcode_to_h264(src, dst, args.crf, args.preset, ffmpeg_bin)

    if args.copy_non_video:
        if args.dry_run:
            print("[INFO] dry_run enabled, non-video files would also be copied")
        else:
            copied, skipped = copy_non_video_files(
                input_root=input_root,
                output_root=output_root,
                pattern=args.pattern,
                overwrite=args.overwrite,
            )
            print(f"[INFO] copied non-video files: {copied}")
            print(f"[INFO] skipped existing non-video files: {skipped}")

    print("[INFO] done")


if __name__ == "__main__":
    main()
