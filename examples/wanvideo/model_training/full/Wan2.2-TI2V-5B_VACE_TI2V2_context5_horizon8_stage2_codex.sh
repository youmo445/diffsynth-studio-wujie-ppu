#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/mnt/data/zsq/DiffSynth-Studio
WAN22_DIR=/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B
WAN_T5=/mnt/data/zsq/Wan_model/Wan2.1-VACE-14B/models_t5_umt5-xxl-enc-bf16.pth
STAGE1_DIT="${WORKDIR}/outputs/Wan2.2-TI2V-5B-TI2V2-context5-horizon8-codex-nonoise/epoch-49.safetensors"

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
cd "${WORKDIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 /usr/local/bin/accelerate launch \
  --config_file "${WORKDIR}/examples/wanvideo/model_training/full/accelerate_config_14B.yaml" \
  "${WORKDIR}/examples/wanvideo/model_training/train_wan22_ti2v2_vace_context5_codex.py" \
  --num_frames 13 \
  --dataset_repeat 1 \
  --model_paths "[
    \"${STAGE1_DIT}\",
    \"${WAN_T5}\",
    \"${WAN22_DIR}/Wan2.2_VAE.pth\"
  ]" \
  --learning_rate 1e-4 \
  --num_epochs 10000 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "${WORKDIR}/outputs/Wan2.2-TI2V-5B-VACE-TI2V2-context5-horizon8-codex-stage2-epoch49-nonoise" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,ray_map_o,ray_map_d" \
  --wan22_vace_in_dim 144 \
  --val_interval 5 \
  --save_epochs 5 \
  --agibot_multiview_base_path "/mnt/data/zsq/Agi2024subset_split" \
  --dataset_context_frames 5 \
  --dataset_horizon_frames 8 \
  --dataset_first_chunk_prob 0.05 \
  --dataset_output_raymap \
  --dataset_raymap_mode image \
  --dataset_traj_radius_mode perspective \
  --dataset_camera_names head hand_left hand_right \
  --dataset_camera_sample_mode all \
  --dataset_val_camera_sample_mode all \
  --ti2v2_low_noise_step_offset 1 \
  --ti2v2_horizon_loss_latent_slots 2
