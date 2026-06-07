#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/mnt/data/zsq/DiffSynth-Studio
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
cd "${WORKDIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 accelerate launch \
  --config_file "${WORKDIR}/examples/wanvideo/model_training/full/accelerate_config_14B.yaml" \
  "${WORKDIR}/examples/wanvideo/model_training/train_wan22_codex.py" \
  --num_frames 9 \
  --dataset_repeat 1 \
  --model_paths '[
    ["/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
    "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
    "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"],
    "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
    "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
  ]' \
  --learning_rate 1e-4 \
  --num_epochs 10000 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "${WORKDIR}/outputs/Wan2.2-TI2V-5B-VACE-wan22-codex-ray-action-perspective-9-applymask" \
  --trainable_models "vace" \
  --extra_inputs "input_image,vace_video,ray_map_o,ray_map_d" \
  --wan22_vace_in_dim 144 \
  --val_interval 5 \
  --save_epochs 5 \
  --dataset AgisubDatasetvacemultiview \
  --agibot_multiview_base_path "/mnt/data/zsq/Agi2024subset_split" \
  --dataset_output_raymap \
  --dataset_raymap_mode image \
  --dataset_traj_radius_mode perspective \
  --dataset_camera_names head hand_left hand_right \
  --dataset_camera_sample_mode all \
  --dataset_val_camera_sample_mode all
