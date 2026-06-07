import os
import types

import numpy as np
import torch
from safetensors.torch import load_file

from diffsynth.models.wan_video_vace_wan22_codex import VaceWan22CodexModel
from diffsynth.pipelines.wan_video_new_wan22_codex import ModelConfig, WanVideoPipeline
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.utils_codex import AgiBotWCDataset4WanControlmultiview

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AgiBotWCDataset4WanControlmultiviewTI2V2Codex(AgiBotWCDataset4WanControlmultiview):
    """VACE multiview dataset with the same context5/horizon8 sampling as TI2V2 stage 1."""

    def __init__(self, *args, context_frames=5, horizon_frames=8, first_chunk_prob=0.05, **kwargs):
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
            f"[AgiBotWCDataset4WanControlmultiviewTI2V2Codex] "
            f"context={self.context_frames} horizon={self.horizon_frames} "
            f"first_chunk_prob={self.first_round_prob} samples={self.total_samples} length={self.length}"
        )

    def _make_frame_ids(self, context_start_idx):
        if np.random.rand() < self.first_round_prob:
            return np.array([0] * self.context_frames + list(range(1, self.horizon_frames + 1)))
        consecutive_ids = np.arange(context_start_idx, context_start_idx + self.num_frames - 1)
        return np.concatenate([[0], consecutive_ids])


def _ti2v2_vace_training_loss(self, **inputs):
    print(f'TI2V2_VACE_5context的training_loss被调用')
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

    input_latents = inputs["input_latents"]
    noise = inputs["noise"]
    if input_latents.shape[2] < 4:
        raise ValueError(f"Wan2.2 TI2V2 VACE expects at least 4 latent slots, got {tuple(input_latents.shape)}.")

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
    horizon_slots = int(inputs.get("ti2v2_horizon_loss_latent_slots", 2))
    if horizon_slots <= 0 or horizon_slots > loss.shape[2]:
        raise ValueError(f"Invalid horizon latent slots {horizon_slots} for loss shape {tuple(loss.shape)}.")
    mask = torch.zeros_like(loss)
    mask[:, :, -horizon_slots:] = 1.0
    print(f'将loss限制在最后{horizon_slots}个latent槽位上')
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
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        wan22_vace_in_dim=144,
        wan22_vace_random_init=False,
        dit_checkpoint_path=None,
        ti2v2_low_noise_step_offset=50,
        ti2v2_horizon_loss_latent_slots=2,
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
            print(f"Initialized Wan2.2 Codex VACE blocks from stage-1 DiT: {len(summaries)} blocks.")

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
            "clean_latent_slots": 2, # 这里先适配预训练的Wan2.2-TI2V-5B
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
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        return self.pipe.training_loss(**models, **inputs)


if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--dit_checkpoint_path", type=str, default=None)
    parser.add_argument("--wan22_vace_in_dim", type=int, default=144)
    parser.add_argument("--wan22_vace_random_init", action="store_true")
    parser.add_argument("--ti2v2_low_noise_step_offset", type=int, default=50)
    parser.add_argument("--ti2v2_horizon_loss_latent_slots", type=int, default=2)
    parser.add_argument("--agibot_multiview_base_path", type=str, default="/mnt/data/zsq/Agi2024subset_split")
    parser.add_argument("--dataset_camera_names", nargs="+", default=["head", "hand_left", "hand_right"])
    parser.add_argument("--dataset_context_frames", type=int, default=5)
    parser.add_argument("--dataset_horizon_frames", type=int, default=8)
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

    val_camera_sample_mode = args.dataset_val_camera_sample_mode or args.dataset_camera_sample_mode
    dataset_kwargs = dict(
        repeat=args.dataset_repeat,
        dataset_type="vace",
        output_raymap=args.dataset_output_raymap,
        raymap_mode=args.dataset_raymap_mode,
        traj_radius_mode=args.dataset_traj_radius_mode,
        camera_names=args.dataset_camera_names,
        context_frames=args.dataset_context_frames,
        horizon_frames=args.dataset_horizon_frames,
        first_chunk_prob=args.dataset_first_chunk_prob,
    )
    dataset = AgiBotWCDataset4WanControlmultiviewTI2V2Codex(
        base_path=os.path.join(args.agibot_multiview_base_path, "train"),
        camera_sample_mode=args.dataset_camera_sample_mode,
        **dataset_kwargs,
    )
    val_dataset = AgiBotWCDataset4WanControlmultiviewTI2V2Codex(
        base_path=os.path.join(args.agibot_multiview_base_path, "val"),
        camera_sample_mode=val_camera_sample_mode,
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
        use_gradient_checkpointing=getattr(args, "use_gradient_checkpointing", False),
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
