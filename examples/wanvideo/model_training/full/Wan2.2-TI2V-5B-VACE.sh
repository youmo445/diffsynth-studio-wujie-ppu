CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 accelerate launch \
  --config_file /mnt/workspace/zsq/DiffSynth-Studio/examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  /mnt/workspace/zsq/DiffSynth-Studio/examples/wanvideo/model_training/train.py \
  --num_frames 9 \
  --dataset_repeat 1 \
  --model_paths '[
    ["/mnt/workspace/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
    "/mnt/workspace/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
    "/mnt/workspace/zsq/Wan_model/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"],
    "/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.1-VACE-14B/epoch-0.safetensors",
    "/mnt/workspace/zsq/Wan_model/Wan2.1-VACE-14B/models_t5_umt5-xxl-enc-bf16.pth",
    "/mnt/workspace/zsq/Wan_model/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 10000 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.2-TI2V-VACE" \
  --trainable_models "vace" \
  --extra_inputs "input_image,vace_video,vace_reference_image" \
  --val_interval 5 \
  --save_epochs 10 \
  --dataset AgisubDatasetvace \