#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/mnt/data/zsq/DiffSynth-Studio
MASTER_PORT=${MASTER_PORT:-29500}
EPISODES_PER_BASE=${EPISODES_PER_BASE:-1000}

export NCCL_SOCKET_IFNAME=net0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export PYTHONMALLOC=malloc

export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
cd "${WORKDIR}"

accelerate launch \
  --config_file "${WORKDIR}/examples/wanvideo/model_training/full/accelerate_config_14B_multinode.yaml" \
  --num_machines 3 \
  --num_processes 48 \
  --machine_rank "${RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  "${WORKDIR}/examples/wanvideo/model_training/train_wan22_ti2v2_vace_arxx5_multi_dataset_codex.py" \
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
  --output_path "${WORKDIR}/outputs/Wan2.2-TI2V-5B-VACE-TI2V2-context5-horizon8-arxx5-0520-moretasks" \
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
  --agibot_multiview_base_path  "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/invert_socks/pack_socks.black.M.invert.200s.20260122.batch.5" "/mnt/data/zsq/stack_blocks_bimanual_0520" \
  --agibot_multiview_recursive_discover \
  --agibot_multiview_episodes_per_base "${EPISODES_PER_BASE}" \
  --agibot_multiview_val_base_path "/mnt/data/zsq/stack_blocks_bimanual_0520" \
  --agibot_multiview_val_tail_episodes 5 \
  --dataset_output_raymap \
  --dataset_raymap_mode image \
  --dataset_traj_radius_mode perspective \
  --dataset_camera_names head left_wrist right_wrist \
  --dataset_camera_sample_mode all \
  --dataset_val_camera_sample_mode all
