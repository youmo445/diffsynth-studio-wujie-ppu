import os

import torch

from diffsynth.pipelines.wan_video_new_wan22_ti2v2_codex import ModelConfig, WanVideoPipeline
from diffsynth.trainers.agibot_wan22_ti2v2_dataset_codex import AgiBotWan22TI2V2Dataset
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
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
            "clean_latent_slots": 2,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "ti2v2_low_noise_step_offset": self.ti2v2_low_noise_step_offset,
            "ti2v2_horizon_loss_latent_slots": self.ti2v2_horizon_loss_latent_slots,
        }
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


if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument(
        "--agibot_ti2v2_base_path",
        type=str,
        default="/mnt/data/zsq/Agi2024subset_split",
        help="Directory containing train/ and val/ AgiBot subset splits.",
    )
    parser.add_argument("--dataset_camera_names", nargs="+", default=["head", "hand_left", "hand_right"])
    parser.add_argument("--dataset_view_height", type=int, default=320)
    parser.add_argument("--dataset_view_width", type=int, default=512)
    parser.add_argument("--dataset_original_hz", type=int, default=30)
    parser.add_argument("--dataset_target_hz", type=int, default=5)
    parser.add_argument("--dataset_sample_stride", type=int, default=1)
    parser.add_argument("--dataset_context_frames", type=int, default=5)
    parser.add_argument("--dataset_horizon_frames", type=int, default=8)
    parser.add_argument("--dataset_prompt", type=str, default="机械臂按照要求移动夹爪执行任务")
    parser.add_argument("--dataset_first_chunk_prob", type=float, default=0.05)
    parser.add_argument("--no_dataset_h264_redirect", action="store_true")
    parser.add_argument("--ti2v2_low_noise_step_offset", type=int, default=50)
    parser.add_argument("--ti2v2_horizon_loss_latent_slots", type=int, default=2)
    args = parser.parse_args()

    dataset_kwargs = dict(
        num_frames=args.num_frames,
        context_frames=args.dataset_context_frames,
        horizon_frames=args.dataset_horizon_frames,
        camera_names=args.dataset_camera_names,
        view_height=args.dataset_view_height,
        view_width=args.dataset_view_width,
        original_hz=args.dataset_original_hz,
        target_hz=args.dataset_target_hz,
        sample_stride=args.dataset_sample_stride,
        repeat=args.dataset_repeat,
        prompt=args.dataset_prompt,
        use_h264_redirect=not args.no_dataset_h264_redirect,
        first_chunk_prob=args.dataset_first_chunk_prob,
    )
    dataset = AgiBotWan22TI2V2Dataset(
        base_path=os.path.join(args.agibot_ti2v2_base_path, "train"),
        **dataset_kwargs,
    )
    val_dataset = AgiBotWan22TI2V2Dataset(
        base_path=os.path.join(args.agibot_ti2v2_base_path, "val"),
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
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        ti2v2_low_noise_step_offset=args.ti2v2_low_noise_step_offset,
        ti2v2_horizon_loss_latent_slots=args.ti2v2_horizon_loss_latent_slots,
    )
    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    launch_training_task(dataset, val_dataset, model, model_logger, args=args)
