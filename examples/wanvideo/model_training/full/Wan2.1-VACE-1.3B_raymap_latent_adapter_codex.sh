CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 accelerate launch \
  --config_file /mnt/workspace/zsq/DiffSynth-Studio/examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  /mnt/workspace/zsq/DiffSynth-Studio/examples/wanvideo/model_training/train_codex.py \
  --num_frames 9 \
  --dataset_repeat 1 \
  --model_paths '[
    "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
    "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/models_t5_umt5-xxl-enc-bf16.pth",
    "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/Wan2.1_VAE.pth"
  ]' \
  --learning_rate 1e-4 \
  --num_epochs 10000 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.1-VACE-1.3B-multiview-raymap-perspective-latentraymap-adapter" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_reference_image,ray_map_o,ray_map_d" \
  --val_interval 5 \
  --save_epochs 5 \
  --dataset AgisubDatasetvacemultiview \
  --dataset_output_raymap \
  --dataset_raymap_mode latent \
  --dataset_traj_radius_mode perspective \
  --enable_vace_latent_raymap_adapter
