import torch, os, json
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset, LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.trainers.utils import AgiBotWCDataset4WanControl
from diffsynth.trainers.utils_codex import AgiBotWCDataset4WanControlmultiview
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        enable_vace_global_cross_attn=False,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(model_id=audio_processor_config.split(":")[0], origin_file_pattern=audio_processor_config.split(":")[1])
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, audio_processor_config=audio_processor_config)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        if enable_vace_global_cross_attn:
            if self.pipe.vace is None:
                raise ValueError("--enable_vace_global_cross_attn requires a VACE model.")
            self.pipe.vace.enable_global_cross_attn(global_context_dim=16)
            if self.pipe.vace2 is not None:
                self.pipe.vace2.enable_global_cross_attn(global_context_dim=16)
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        if enable_vace_global_cross_attn and "vace_global_reference_images" not in self.extra_inputs:
            self.extra_inputs.append("vace_global_reference_images")
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval in epochs")
    parser.add_argument("--dataset",type=str,default="RLinfNpyDataset",help="Dataset type for training")
    parser.add_argument("--dataset_output_raymap", action="store_true", help="Output ray_map_o and ray_map_d from codex multiview dataset")
    parser.add_argument(
        "--dataset_camera_names",
        nargs="+",
        default=None,
        help="Camera names for codex multiview dataset; default keeps all dataset cameras",
    )
    parser.add_argument(
        "--dataset_camera_sample_mode",
        type=str,
        default="all",
        choices=("all", "random_one", "cycle_one"),
        help="Use all cameras, randomly sample one camera per item, or cycle one camera per item",
    )
    parser.add_argument(
        "--dataset_val_camera_sample_mode",
        type=str,
        default=None,
        choices=("all", "random_one", "cycle_one"),
        help="Camera sampling mode for validation; default follows --dataset_camera_sample_mode",
    )
    parser.add_argument(
        "--dataset_traj_radius_mode",
        type=str,
        default="constant",
        choices=("constant", "perspective", "depth_norm"),
        help="Trajectory point radius mode for codex multiview dataset",
    )
    parser.add_argument(
        "--dataset_raymap_mode",
        type=str,
        default="image",
        choices=("image", "latent"),
        help="Use image raymaps encoded by VAE or latent-resolution raw raymaps from the dataset.",
    )
    parser.add_argument(
        "--enable_vace_global_cross_attn",
        action="store_true",
        help="Encode all-view reference images with Wan VAE and let VACE tokens cross-attend them.",
    )
    args = parser.parse_args()
    val_camera_sample_mode = args.dataset_val_camera_sample_mode or args.dataset_camera_sample_mode

    if args.dataset == "AgisubDataset":
        dataset = AgiBotWCDataset4WanControl(
            base_path="/mnt/workspace/zsq/Agibotsubset/train",
            num_frames = args.num_frames,
            context_length = 1,
            repeat=args.dataset_repeat,
            dataset_type = 'control'
        )
        val_dataset = AgiBotWCDataset4WanControl(
            base_path="/mnt/workspace/zsq/Agibotsubset/val",
            num_frames = args.num_frames,
            context_length = 1,
            repeat=args.dataset_repeat,
            dataset_type = 'control'
        )
    elif args.dataset == "AgisubDatasetvace":
        dataset = AgiBotWCDataset4WanControl(
            base_path="/mnt/workspace/zsq/Agibotsubset/train",
            num_frames = args.num_frames,
            context_length = 1,
            repeat=args.dataset_repeat,
            dataset_type = 'vace'
        )
        val_dataset = AgiBotWCDataset4WanControl(
            base_path="/mnt/workspace/zsq/Agibotsubset/val",
            num_frames = args.num_frames,
            context_length = 1,
            repeat=args.dataset_repeat,
            dataset_type = 'vace'
        )
    elif args.dataset == "AgisubDatasetvacemultiview":
        dataset = AgiBotWCDataset4WanControlmultiview(
            base_path="/mnt/workspace/zsq/Agi2024subset_split/train",
            num_frames = args.num_frames,
            context_length = 1,
            repeat=args.dataset_repeat,
            dataset_type = 'vace',
            output_raymap=args.dataset_output_raymap,
            raymap_mode=args.dataset_raymap_mode,
            output_global_reference=args.enable_vace_global_cross_attn,
            traj_radius_mode=args.dataset_traj_radius_mode,
            camera_names=args.dataset_camera_names,
            camera_sample_mode=args.dataset_camera_sample_mode,
        )
        val_dataset = AgiBotWCDataset4WanControlmultiview(
            base_path="/mnt/workspace/zsq/Agi2024subset_split/val",
            num_frames = args.num_frames,
            context_length = 1,
            repeat=args.dataset_repeat,
            dataset_type = 'vace',
            output_raymap=args.dataset_output_raymap,
            raymap_mode=args.dataset_raymap_mode,
            output_global_reference=args.enable_vace_global_cross_attn,
            traj_radius_mode=args.dataset_traj_radius_mode,
            camera_names=args.dataset_camera_names,
            camera_sample_mode=val_camera_sample_mode,
        )
    else:
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
            ),
            special_operator_map={
                "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
                "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            }
        )    
    # ----------------------

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        enable_vace_global_cross_attn=args.enable_vace_global_cross_attn,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, val_dataset, model, model_logger, args=args)
