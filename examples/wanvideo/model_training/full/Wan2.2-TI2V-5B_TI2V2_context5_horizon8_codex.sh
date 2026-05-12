#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/mnt/data/zsq/DiffSynth-Studio
WAN22_DIR=/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
cd "${WORKDIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 /usr/local/bin/accelerate launch \
  --config_file "${WORKDIR}/examples/wanvideo/model_training/full/accelerate_config_14B.yaml" \
  "${WORKDIR}/examples/wanvideo/model_training/train_wan22_ti2v2_context5_codex.py" \
  --num_frames 13 \
  --dataset_repeat 1 \
  --model_paths '[
    ["/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
    "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
    "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"],
    "/mnt/data/zsq/Wan_model/Wan2.1-VACE-14B/models_t5_umt5-xxl-enc-bf16.pth",
    "/mnt/data/zsq/Wan_model/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 10000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${WORKDIR}/outputs/Wan2.2-TI2V-5B-TI2V2-context5-horizon8-codex-nonoise" \
  --trainable_models "dit" \
  --val_interval 5 \
  --save_epochs 5 \
  --agibot_ti2v2_base_path "/mnt/data/zsq/Agi2024subset_split" \
  --dataset_camera_names head hand_left hand_right \
  --dataset_view_height 320 \
  --dataset_view_width 512 \
  --dataset_context_frames 5 \
  --dataset_horizon_frames 8 \
  --dataset_original_hz 30 \
  --dataset_target_hz 5 \
  --dataset_sample_stride 1 \
  --ti2v2_low_noise_step_offset 1 \
  --ti2v2_horizon_loss_latent_slots 2
