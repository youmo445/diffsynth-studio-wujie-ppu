#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/mnt/data/zsq/DiffSynth-Studio
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
cd "${WORKDIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, accelerate launch \
  --config_file "${WORKDIR}/examples/wanvideo/model_training/full/accelerate_config_14B.yaml" \
  "${WORKDIR}/examples/wanvideo/model_training/train_wan22_ti2v2_vace_arxx5_context5_horizon16_codex.py" \
  --num_frames 13 \
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
  --output_path "${WORKDIR}/outputs/Wan2.2-TI2V-5B-VACE-TI2V2-context5-horizon8-arxx5-0520-t1jiazao" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,ray_map_o,ray_map_d" \
  --wan22_vace_in_dim 144 \
  --ti2v2_low_noise_step_offset 1 \
  --ti2v2_horizon_loss_latent_slots 2 \
  --val_interval 5 \
  --save_epochs 10 \
  --dataset Arxx5Dataset4Wancontrolmultiview \
  --dataset_context_frames 5 \
  --dataset_horizon_frames 8 \
  --dataset_first_chunk_prob 0.05 \
  --agibot_multiview_base_path  "/mnt/data/zsq/arx_policy_rollout_0520" "/mnt/data/zsq/stack_blocks_bimanual_0520" \
  --agibot_multiview_val_base_path "/mnt/data/zsq/stack_blocks_bimanual_0520" \
  --agibot_multiview_val_tail_episodes 5 \
  --dataset_output_raymap \
  --dataset_raymap_mode image \
  --dataset_traj_radius_mode perspective \
  --dataset_camera_names head left_wrist right_wrist \
  --dataset_camera_sample_mode all \
  --dataset_val_camera_sample_mode all
