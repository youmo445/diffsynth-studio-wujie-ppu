import os
from dataclasses import dataclass

import numpy as np
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _Episode:
    key: str
    video_dir: str
    frame_count: int


class AgiBotWan22TI2V2Dataset(Dataset):
    """Three-view AgiBot video dataset for Wan2.2 TI2V2 prefix training.

    Each sample has 5 context frames + 8 horizon frames by default. The first
    context frame is always the first frame of the episode. Normal samples use
    the four frames before the horizon as the rest of the context. A small
    first-chunk probability repeats the episode first frame for all context
    frames to match inference on the first generated chunk.
    """

    def __init__(
        self,
        base_path,
        num_frames=13,
        context_frames=5,
        horizon_frames=8,
        camera_names=("head", "hand_left", "hand_right"),
        view_height=320,
        view_width=512,
        original_hz=30,
        target_hz=5,
        sample_stride=1,
        repeat=1,
        prompt="机械臂按照要求移动夹爪执行任务",
        use_h264_redirect=True,
        first_chunk_prob=0.05,
    ):
        if num_frames != context_frames + horizon_frames:
            raise ValueError(
                f"num_frames must equal context_frames + horizon_frames, got "
                f"{num_frames} != {context_frames} + {horizon_frames}."
            )
        if original_hz % target_hz != 0:
            raise ValueError("original_hz must be divisible by target_hz.")

        self.base_path = os.path.abspath(base_path)
        self.video_base_path = self._redirect_h264(self.base_path) if use_h264_redirect else self.base_path
        self.num_frames = int(num_frames)
        self.context_frames = int(context_frames)
        self.horizon_frames = int(horizon_frames)
        self.camera_names = tuple(camera_names)
        self.view_height = int(view_height)
        self.view_width = int(view_width)
        self.downsample_step = original_hz // target_hz
        self.sample_stride = int(sample_stride)
        self.repeat = int(repeat)
        self.prompt = prompt
        self.load_from_cache = False
        self.first_chunk_prob = float(first_chunk_prob)
        if self.context_frames < 2:
            raise ValueError("context_frames must be at least 2 for episode-anchor sampling.")
        if not 0.0 <= self.first_chunk_prob <= 1.0:
            raise ValueError(f"first_chunk_prob must be in [0, 1], got {self.first_chunk_prob}.")

        self.episodes = self._discover_episodes()
        self.sample_indices = self._build_sample_indices()
        if not self.sample_indices:
            raise RuntimeError(f"No valid {num_frames}-frame samples found under {self.video_base_path}.")
        print(
            f"[AgiBotWan22TI2V2Dataset] base={self.base_path} video_base={self.video_base_path} "
            f"episodes={len(self.episodes)} samples={len(self.sample_indices)} repeat={self.repeat} "
            f"first_chunk_prob={self.first_chunk_prob}"
        )

    def _redirect_h264(self, path):
        marker = "Agi2024subset_split"
        if marker not in path or "Agi2024subset_split_h264" in path:
            return path
        redirected = path.replace(marker, f"{marker}_h264", 1)
        return redirected if os.path.isdir(redirected) else path

    def _video_path(self, video_dir, camera_name):
        return os.path.join(video_dir, f"{camera_name}_color.mp4")

    def _discover_episodes(self):
        obs_root = os.path.join(self.video_base_path, "observations")
        if not os.path.isdir(obs_root):
            raise FileNotFoundError(f"Missing observations directory: {obs_root}")

        episodes = []
        for task in sorted(os.listdir(obs_root)):
            task_dir = os.path.join(obs_root, task)
            if not os.path.isdir(task_dir):
                continue
            for ep in sorted(os.listdir(task_dir)):
                video_dir = os.path.join(task_dir, ep, "videos")
                if not os.path.isdir(video_dir):
                    continue
                paths = [self._video_path(video_dir, cam) for cam in self.camera_names]
                if not all(os.path.exists(path) for path in paths):
                    continue
                frame_count = min(self._estimate_frame_count(path) for path in paths)
                if frame_count > 0:
                    episodes.append(_Episode(key=f"{task}-{ep}", video_dir=video_dir, frame_count=frame_count))
        return episodes

    def _estimate_frame_count(self, path):
        vr = VideoReader(path, ctx=cpu(0))
        frame_count = len(vr)
        del vr
        return frame_count

    def _build_sample_indices(self):
        indices = []
        context_history = self.context_frames - 1
        min_horizon_start_ds = self.context_frames
        horizon_span = (self.horizon_frames - 1) * self.downsample_step
        for ep_id, episode in enumerate(self.episodes):
            max_horizon_start = episode.frame_count - 1 - horizon_span
            if max_horizon_start < min_horizon_start_ds * self.downsample_step:
                continue
            max_horizon_start_ds = max_horizon_start // self.downsample_step
            for horizon_start_ds in range(min_horizon_start_ds, max_horizon_start_ds + 1, self.sample_stride):
                indices.append((ep_id, horizon_start_ds))
        return indices

    def __len__(self):
        return len(self.sample_indices) * self.repeat

    def _read_window(self, path, raw_indices):
        vr = VideoReader(path, ctx=cpu(0))
        safe_indices = [min(int(idx), len(vr) - 1) for idx in raw_indices]
        frames = vr.get_batch(safe_indices).asnumpy()
        del vr
        return [
            np.asarray(Image.fromarray(frame).resize((self.view_width, self.view_height), Image.BILINEAR))
            for frame in frames
        ]

    def __getitem__(self, idx):
        ep_id, horizon_start_ds = self.sample_indices[idx % len(self.sample_indices)]
        episode = self.episodes[ep_id]
        if np.random.random() < self.first_chunk_prob:
            raw_indices = [0] * self.context_frames
            raw_indices += [i * self.downsample_step for i in range(1, self.horizon_frames + 1)]
            sample_type = "first_chunk"
        else:
            raw_indices = [0]
            context_start_ds = horizon_start_ds - (self.context_frames - 1)
            raw_indices += [
                (context_start_ds + i) * self.downsample_step
                for i in range(self.context_frames - 1)
            ]
            raw_indices += [
                (horizon_start_ds + i) * self.downsample_step
                for i in range(self.horizon_frames)
            ]
            sample_type = "rolling"

        per_camera = [
            self._read_window(self._video_path(episode.video_dir, cam), raw_indices)
            for cam in self.camera_names
        ]
        video = []
        for frame_id in range(self.num_frames):
            stacked = np.concatenate([frames[frame_id] for frames in per_camera], axis=0)
            video.append(Image.fromarray(stacked))

        return {
            "video": video,
            "prompt": self.prompt,
            "episode_key": episode.key,
            "horizon_start_ds": horizon_start_ds,
            "sample_type": sample_type,
        }
