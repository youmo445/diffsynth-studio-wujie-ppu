"""
Extract the first frame of a specified LeRobot v2.1 episode and save a
height-concatenated three-view image.

Expected dataset layout:
    dataset_root/
        data/chunk-xxx/episode_000123.parquet
        videos/chunk-xxx/observation.images.head/episode_000123.mp4
        videos/chunk-xxx/observation.images.left_wrist/episode_000123.mp4
        videos/chunk-xxx/observation.images.right_wrist/episode_000123.mp4

Example:
    cd /mnt/workspace/zsq/DiffSynth-Studio
    python examples/wanvideo/model_inference/extract_lerobot_episode_first_frame.py \
        --dataset_root /mnt/data/zsq/agx \
        --episode_index 10 \
        --output_path /mnt/workspace/zsq/DiffSynth-Studio/outputs/episode_000010_first_frame.png
"""

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_CAMERAS = ("head", "left_wrist", "right_wrist")


def discover_episodes(dataset_root, camera_names):
    dataset_root = Path(dataset_root)
    data_dir = dataset_root / "data"
    videos_dir = dataset_root / "videos"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    episodes = []
    for parquet_path in sorted(data_dir.glob("chunk-*/episode_*.parquet")):
        match = re.search(r"episode_(\d+)\.parquet$", parquet_path.name)
        if match is None:
            continue

        episode_index = int(match.group(1))
        chunk = parquet_path.parent.name
        video_paths = {
            cam: videos_dir / chunk / f"observation.images.{cam}" / f"episode_{episode_index:06d}.mp4"
            for cam in camera_names
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
    return episodes


def select_episode(episodes, episode_index=None, episode_pos=0):
    if episode_index is not None:
        matched = [episode for episode in episodes if episode["episode_index"] == episode_index]
        if not matched:
            raise RuntimeError(f"episode_index={episode_index} not found")
        return matched[0]

    if not episodes:
        raise RuntimeError("No episodes found")

    if episode_pos < 0 or episode_pos >= len(episodes):
        raise RuntimeError(f"episode_pos out of range: {episode_pos}, total={len(episodes)}")
    return episodes[episode_pos]


def read_first_video_frame(video_path):
    import av

    with av.open(str(video_path)) as container:
        stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if stream is None:
            raise RuntimeError(f"No video stream found in {video_path}")

        for frame in container.decode(stream):
            return frame.to_ndarray(format="rgb24")

    raise RuntimeError(f"No frames found in {video_path}")


def resolve_target_size(frames, view_height=None, view_width=None):
    if view_height is not None and view_width is not None:
        return int(view_height), int(view_width)
    if view_height is None and view_width is None:
        first_h, first_w = frames[0].shape[:2]
        return int(first_h), int(first_w)
    raise ValueError("--view_height and --view_width must be set together")


def build_concat_image(video_paths, camera_names, view_height=None, view_width=None):
    raw_frames = [read_first_video_frame(video_paths[cam]) for cam in camera_names]
    out_h, out_w = resolve_target_size(raw_frames, view_height, view_width)

    resized_frames = []
    for frame in raw_frames:
        pil_image = Image.fromarray(frame)
        if pil_image.size != (out_w, out_h):
            pil_image = pil_image.resize((out_w, out_h), Image.BILINEAR)
        resized_frames.append(np.asarray(pil_image, dtype=np.uint8))

    concat_frame = np.concatenate(resized_frames, axis=0)
    return Image.fromarray(concat_frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/mnt/workspace/zsq/arx_policy_rollout_new_continue")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--episode_pos", type=int, default=0)
    parser.add_argument("--camera_names", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--view_height", type=int, default=160)
    parser.add_argument("--view_width", type=int, default=224)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    episodes = discover_episodes(args.dataset_root, args.camera_names)
    print(f"[INFO] discovered episodes: {len(episodes)}")
    episode = select_episode(episodes, args.episode_index, args.episode_pos)
    print(
        f"[INFO] selected episode_index={episode['episode_index']} "
        f"chunk={episode['chunk']} parquet={episode['parquet_path']}"
    )

    concat_image = build_concat_image(
        episode["video_paths"],
        args.camera_names,
        args.view_height,
        args.view_width,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_image.save(output_path)
    print(f"[DONE] saved image to {output_path}")


if __name__ == "__main__":
    main()