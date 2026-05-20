#!/usr/bin/env python3
import os
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from diffsynth.models.wan_video_vace_wan22_codex import VaceWan22CodexModel
from diffsynth.pipelines.wan_video_new_wan22_codex import ModelConfig, WanVideoPipeline
from diffsynth.trainers.arxx5_dataset4_wancontrolmultiview import Arxx5Dataset4Wancontrolmultiview, load_json
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Arxx5Dataset4WancontrolmultiviewTI2V2Codex(Arxx5Dataset4Wancontrolmultiview):
    """ARXX5 VACE multiview dataset with TI2V2 context5/horizon16 sampling."""

    def __init__(
        self,
        *args,
        context_frames=5,
        horizon_frames=16,
        first_chunk_prob=0.05,
        precomputed_episodes=None,
        **kwargs,
    ):
        self._precomputed_episodes = precomputed_episodes
        num_frames = int(context_frames) + int(horizon_frames)
        super().__init__(
            *args,
            num_frames=num_frames,
            context_length=int(context_frames),
            first_round_prob=float(first_chunk_prob),
            **kwargs,
        )
        self.context_frames = int(context_frames)
        self.horizon_frames = int(horizon_frames)
        self._rebuild_ti2v2_indices()

    def _discover_episodes(self):
        if self._precomputed_episodes is not None:
            return list(self._precomputed_episodes)
        return super()._discover_episodes()

    def _rebuild_ti2v2_indices(self):
        self.sample_indices = []
        consecutive_count = self.num_frames - 1
        for ep_idx, info in enumerate(self.episode_info):
            max_context_start = int(info["T_ds"]) - consecutive_count
            for context_start in range(1, max_context_start + 1, self.stride):
                self.sample_indices.append((ep_idx, context_start))
        self.total_samples = len(self.sample_indices)
        self.length = self.total_samples * self.repeat
        print(
            f"[Arxx5Dataset4WancontrolmultiviewTI2V2Codex] "
            f"base={self.base_path} context={self.context_frames} horizon={self.horizon_frames} "
            f"first_chunk_prob={self.first_round_prob} samples={self.total_samples} length={self.length}"
        )

    def _make_frame_ids(self, context_start_idx):
        if np.random.rand() < self.first_round_prob:
            return np.array([0] * self.context_frames + list(range(1, self.horizon_frames + 1)), dtype=np.int64)
        consecutive_ids = np.arange(context_start_idx, context_start_idx + self.num_frames - 1, dtype=np.int64)
        return np.concatenate([[0], consecutive_ids])


def _ti2v2_vace_training_loss(self, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

    input_latents = inputs["input_latents"]
    noise = inputs["noise"]
    if input_latents.shape[2] < 2:
        raise ValueError(f"Wan2.2 TI2V2 VACE expects at least 2 latent slots, got {tuple(input_latents.shape)}.")

    latents = self.scheduler.add_noise(input_latents, noise, timestep)
    latents[:, :, 0:1] = input_latents[:, :, 0:1]

    max_low_noise_step_offset = int(inputs.get("ti2v2_low_noise_step_offset", 50))
    actual_low_noise_step_offset = int(torch.randint(0, max_low_noise_step_offset + 1, (1,)).item())
    if actual_low_noise_step_offset == 0:
        latents[:, :, 1:2] = input_latents[:, :, 1:2]
    else:
        low_noise_idx = torch.tensor([max(0, len(self.scheduler.timesteps) - actual_low_noise_step_offset)])
        low_noise_timestep = self.scheduler.timesteps[low_noise_idx].to(dtype=self.torch_dtype, device=self.device)
        latents[:, :, 1:2] = self.scheduler.add_noise(
            input_latents[:, :, 1:2],
            noise[:, :, 1:2],
            low_noise_timestep,
        )
    inputs["latents"] = latents

    training_target = self.scheduler.training_target(input_latents, noise, timestep)
    noise_pred = self.model_fn(**inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction="none")
    horizon_slots = int(inputs.get("ti2v2_horizon_loss_latent_slots", 4))
    if horizon_slots <= 0 or horizon_slots > loss.shape[2]:
        raise ValueError(f"Invalid horizon latent slots {horizon_slots} for loss shape {tuple(loss.shape)}.")
    mask = torch.zeros_like(loss)
    mask[:, :, -horizon_slots:] = 1.0
    loss = (loss * mask).sum() / mask.sum().clamp_min(1)
    loss = loss * self.scheduler.training_weight(timestep)
    return loss


def _load_dit_checkpoint(dit, checkpoint_path):
    state_dict = load_file(checkpoint_path, device="cpu")
    prefixes = ("pipe.dit.", "dit.")
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        cleaned[new_key] = value
    incompatible = dit.load_state_dict(cleaned, strict=False)
    print(
        f"Loaded stage-1 DiT checkpoint: {checkpoint_path}; "
        f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
    )
    if incompatible.missing_keys:
        print("First missing keys:", incompatible.missing_keys[:10])
    if incompatible.unexpected_keys:
        print("First unexpected keys:", incompatible.unexpected_keys[:10])


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        wan22_vace_in_dim=144,
        wan22_vace_random_init=False,
        dit_checkpoint_path=None,
        ti2v2_low_noise_step_offset=50,
        ti2v2_horizon_loss_latent_slots=4,
    ):
        super().__init__()
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(
                model_id=audio_processor_config.split(":")[0],
                origin_file_pattern=audio_processor_config.split(":")[1],
            )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            audio_processor_config=audio_processor_config,
        )
        self.pipe.training_loss = types.MethodType(_ti2v2_vace_training_loss, self.pipe)

        if dit_checkpoint_path:
            _load_dit_checkpoint(self.pipe.dit, dit_checkpoint_path)

        created_vace = getattr(self.pipe, "vace", None) is None
        if created_vace:
            self.pipe.vace = VaceWan22CodexModel(vace_in_dim=wan22_vace_in_dim).to(dtype=self.pipe.torch_dtype)
        if created_vace and not wan22_vace_random_init:
            summaries = self.pipe.vace.init_from_dit(self.pipe.dit)
            print(f"Initialized Wan2.2 Codex VACE blocks from DiT: {len(summaries)} blocks.")

        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.ti2v2_low_noise_step_offset = ti2v2_low_noise_step_offset
        self.ti2v2_horizon_loss_latent_slots = ti2v2_horizon_loss_latent_slots

    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "fuse_vae_embedding_in_latents": True,
            "clean_latent_slots": 1,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "ti2v2_low_noise_step_offset": self.ti2v2_low_noise_step_offset,
            "ti2v2_horizon_loss_latent_slots": self.ti2v2_horizon_loss_latent_slots,
        }

        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit,
                self.pipe,
                inputs_shared,
                inputs_posi,
                inputs_nega,
            )
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        return self.pipe.training_loss(**models, **inputs)


def discover_episode_indices(base_path: str):
    data_dir = Path(base_path) / "data"
    indices = []
    for parquet_path in sorted(data_dir.glob("chunk-*/episode_*.parquet")):
        try:
            indices.append(int(parquet_path.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(set(indices))


def tail_episode_indices(base_path: str, tail_count: int):
    if tail_count <= 0:
        return None
    indices = discover_episode_indices(base_path)
    if len(indices) < tail_count:
        raise ValueError(f"Only found {len(indices)} episodes in {base_path}, cannot take last {tail_count}")
    return indices[-tail_count:]


def is_lerobot_root(path: Path):
    return (path / "meta" / "info.json").is_file() and (path / "data").is_dir() and (path / "videos").is_dir()


def iter_lerobot_roots(base_path: str, recursive: bool):
    base = Path(base_path)
    if is_lerobot_root(base):
        yield base
        return
    if not recursive:
        yield base
        return
    prune_names = {
        ".git",
        "__pycache__",
        "data",
        "videos",
        "depth",
        "parameters",
        "proprio_stats",
        "tensorboard",
    }
    for dirpath, dirnames, _ in os.walk(base):
        path = Path(dirpath)
        if is_lerobot_root(path):
            yield path
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if name not in prune_names]


def discover_lerobot_roots(base_path: str, recursive: bool):
    roots = list(iter_lerobot_roots(base_path, recursive))
    if not roots:
        raise FileNotFoundError(f"No LeRobot v2.1 dataset roots found under {base_path}")
    return roots


def format_lerobot_path(root: Path, template: str, episode_index: int, chunks_size: int, video_key: str = None):
    kwargs = {
        "episode_index": int(episode_index),
        "episode_chunk": int(episode_index) // int(chunks_size),
    }
    if video_key is not None:
        kwargs["video_key"] = video_key
    return root / template.format(**kwargs)


def make_episode_record(root: Path, episode_index: int, camera_names, info_json: dict = None):
    info_json = info_json or {}
    chunks_size = int(info_json.get("chunks_size", 1000))
    data_template = info_json.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_template = info_json.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    parquet_path = format_lerobot_path(root, data_template, episode_index, chunks_size)
    if not parquet_path.exists():
        return None
    video_paths = {}
    for cam in camera_names:
        video_key = f"observation.images.{cam}"
        video_path = format_lerobot_path(root, video_template, episode_index, chunks_size, video_key=video_key)
        if not video_path.exists():
            return None
        video_paths[cam] = video_path
    return {"episode_index": int(episode_index), "parquet_path": parquet_path, "video_paths": video_paths}


def take_lerobot_episode_records(root: Path, max_episodes: int, camera_names, exclude_episode_indices=None):
    max_episodes = int(max_episodes)
    if max_episodes <= 0:
        return None
    exclude_indices = {int(idx) for idx in (exclude_episode_indices or [])}
    info_json = load_json(root / "meta" / "info.json")
    records = []
    total_episodes = int(info_json.get("total_episodes", 0) or 0)
    if total_episodes > 0:
        for episode_index in range(total_episodes):
            if episode_index in exclude_indices:
                continue
            record = make_episode_record(root, episode_index, camera_names, info_json)
            if record is None:
                continue
            records.append(record)
            if len(records) >= max_episodes:
                return records
        return records

    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    for chunk_entry in sorted(os.scandir(data_dir), key=lambda entry: entry.name):
        if not chunk_entry.is_dir() or not chunk_entry.name.startswith("chunk-"):
            continue
        with os.scandir(chunk_entry.path) as parquet_entries:
            for parquet_entry in parquet_entries:
                if not parquet_entry.is_file() or not parquet_entry.name.startswith("episode_") or not parquet_entry.name.endswith(".parquet"):
                    continue
                try:
                    episode_index = int(Path(parquet_entry.name).stem.split("_")[-1])
                except ValueError:
                    continue
                if episode_index in exclude_indices:
                    continue
                record = make_episode_record(root, episode_index, camera_names, info_json)
                if record is None:
                    continue
                records.append(record)
                if len(records) >= max_episodes:
                    return records
    return records


def set_load_from_cache(dataset):
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        dataset.load_from_cache = all(getattr(ds, "load_from_cache", False) for ds in dataset.datasets)
    else:
        dataset.load_from_cache = getattr(dataset, "load_from_cache", False)
    return dataset


if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval in epochs")
    parser.add_argument("--dataset", type=str, default="Arxx5Dataset4Wancontrolmultiview")
    parser.add_argument("--dit_checkpoint_path", type=str, default=None)
    parser.add_argument("--wan22_vace_in_dim", type=int, default=144)
    parser.add_argument("--wan22_vace_random_init", action="store_true")
    parser.add_argument("--ti2v2_low_noise_step_offset", type=int, default=50)
    parser.add_argument("--ti2v2_horizon_loss_latent_slots", type=int, default=4)
    parser.add_argument(
        "--agibot_multiview_base_path",
        type=str,
        nargs="+",
        default=["/mnt/data/zsq/agx"],
        help="One or more base directories for ARXX5 multiview VACE training.",
    )
    parser.add_argument(
        "--agibot_multiview_val_base_path",
        type=str,
        default=None,
        help="Validation base directory. Defaults to the first train base path.",
    )
    parser.add_argument(
        "--agibot_multiview_val_tail_episodes",
        type=int,
        default=0,
        help="Use the last N episodes from the validation base path as validation.",
    )
    parser.add_argument(
        "--agibot_multiview_recursive_discover",
        action="store_true",
        help="Recursively discover LeRobot v2.1 dataset roots under each training base path.",
    )
    parser.add_argument(
        "--agibot_multiview_episodes_per_base",
        type=int,
        default=0,
        help="Use at most this many episodes from each user-provided base path. <=0 means use all.",
    )
    parser.add_argument("--dataset_camera_names", nargs="+", default=["head", "left_wrist", "right_wrist"])
    parser.add_argument("--dataset_context_frames", type=int, default=5)
    parser.add_argument("--dataset_horizon_frames", type=int, default=16)
    parser.add_argument("--dataset_first_chunk_prob", type=float, default=0.05)
    parser.add_argument("--dataset_output_raymap", action="store_true")
    parser.add_argument("--dataset_raymap_mode", type=str, default="image", choices=("image", "latent"))
    parser.add_argument(
        "--dataset_traj_radius_mode",
        type=str,
        default="perspective",
        choices=("constant", "perspective", "depth_norm"),
    )
    parser.add_argument("--dataset_camera_sample_mode", type=str, default="all", choices=("all", "random_one", "cycle_one"))
    parser.add_argument("--dataset_val_camera_sample_mode", type=str, default=None, choices=("all", "random_one", "cycle_one"))
    args = parser.parse_args()

    if args.dataset != "Arxx5Dataset4Wancontrolmultiview":
        raise ValueError("This TI2V2 VACE entry only supports --dataset Arxx5Dataset4Wancontrolmultiview.")
    expected_frames = int(args.dataset_context_frames) + int(args.dataset_horizon_frames)
    if int(args.num_frames) != expected_frames:
        raise ValueError(f"--num_frames must equal context+horizon ({expected_frames}), got {args.num_frames}.")

    val_camera_sample_mode = args.dataset_val_camera_sample_mode or args.dataset_camera_sample_mode
    dataset_kwargs = dict(
        repeat=args.dataset_repeat,
        dataset_type="vace",
        output_raymap=args.dataset_output_raymap,
        raymap_mode=args.dataset_raymap_mode,
        traj_radius_mode=args.dataset_traj_radius_mode,
        camera_names=args.dataset_camera_names,
        target_hz=30,
        context_frames=args.dataset_context_frames,
        horizon_frames=args.dataset_horizon_frames,
        first_chunk_prob=args.dataset_first_chunk_prob,
    )
    val_base_path = args.agibot_multiview_val_base_path or args.agibot_multiview_base_path[0]
    val_episode_indices = tail_episode_indices(val_base_path, args.agibot_multiview_val_tail_episodes)

    train_datasets = []
    val_base_abs = os.path.abspath(val_base_path)
    for base_id, base_path in enumerate(args.agibot_multiview_base_path):
        base_datasets = []
        root_count = 0
        selected_episode_count = 0
        remaining_episodes = int(args.agibot_multiview_episodes_per_base)
        limit_episodes = remaining_episodes > 0
        for root in iter_lerobot_roots(base_path, args.agibot_multiview_recursive_discover):
            root_count += 1
            root_str = str(root)
            exclude_episode_indices = val_episode_indices if os.path.abspath(root_str) == val_base_abs else None
            precomputed_episodes = None
            if limit_episodes:
                precomputed_episodes = take_lerobot_episode_records(
                    root,
                    remaining_episodes,
                    args.dataset_camera_names,
                    exclude_episode_indices=exclude_episode_indices,
                )
                if not precomputed_episodes:
                    print(f"[MultiDataset] skip root={root_str}, no valid selected episodes")
                    continue
                selected_episode_count += len(precomputed_episodes)
                remaining_episodes -= len(precomputed_episodes)
            base_datasets.append(
                Arxx5Dataset4WancontrolmultiviewTI2V2Codex(
                    base_path=root_str,
                    camera_sample_mode=args.dataset_camera_sample_mode,
                    precomputed_episodes=precomputed_episodes,
                    exclude_episode_indices=exclude_episode_indices,
                    **dataset_kwargs,
                )
            )
            if limit_episodes and remaining_episodes <= 0:
                break
        if not base_datasets:
            raise FileNotFoundError(f"No usable LeRobot v2.1 dataset roots found under {base_path}")
        if limit_episodes:
            print(
                f"[MultiDataset] base_path={base_path} scanned_roots={root_count} "
                f"selected_episodes={selected_episode_count}/{args.agibot_multiview_episodes_per_base}"
            )
        else:
            print(f"[MultiDataset] base_path={base_path} discovered_roots={root_count}")
        base_dataset = base_datasets[0] if len(base_datasets) == 1 else torch.utils.data.ConcatDataset(base_datasets)
        base_dataset = set_load_from_cache(base_dataset)
        train_datasets.append(base_dataset)
    if len(train_datasets) == 1:
        dataset = set_load_from_cache(train_datasets[0])
    else:
        dataset = torch.utils.data.ConcatDataset(train_datasets)
        dataset = set_load_from_cache(dataset)

    val_dataset = Arxx5Dataset4WancontrolmultiviewTI2V2Codex(
        base_path=val_base_path,
        camera_sample_mode=val_camera_sample_mode,
        episode_indices=val_episode_indices,
        **dataset_kwargs,
    )

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=getattr(args, "use_gradient_checkpointing", True),
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        wan22_vace_in_dim=args.wan22_vace_in_dim,
        wan22_vace_random_init=args.wan22_vace_random_init,
        dit_checkpoint_path=args.dit_checkpoint_path,
        ti2v2_low_noise_step_offset=args.ti2v2_low_noise_step_offset,
        ti2v2_horizon_loss_latent_slots=args.ti2v2_horizon_loss_latent_slots,
    )
    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    launch_training_task(dataset, val_dataset, model, model_logger, args=args)
