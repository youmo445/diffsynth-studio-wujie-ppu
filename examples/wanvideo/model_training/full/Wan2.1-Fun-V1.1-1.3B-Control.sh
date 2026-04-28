CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 accelerate launch \
  --config_file /mnt/workspace/zsq/DiffSynth-Studio/examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  /mnt/workspace/zsq/DiffSynth-Studio/examples/wanvideo/model_training/train.py \
  --num_frames 9 \
  --dataset_repeat 1 \
  --model_paths '[
  "/mnt/workspace/zsq/Wan_model/Wan2.1-Fun-1.3B-Control/diffusion_pytorch_model.safetensors",
  "/mnt/workspace/zsq/Wan_model/Wan2.1-Fun-1.3B-Control-deprated/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
  "/mnt/workspace/zsq/Wan_model/Wan2.1-Fun-1.3B-Control-deprated/models_t5_umt5-xxl-enc-bf16.pth",
  "/mnt/workspace/zsq/Wan_model/Wan2.1-Fun-1.3B-Control-deprated/Wan2.1_VAE.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 10000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/mnt/workspace/zsq/DiffSynth-Studio/outputs/Wan2.1-Fun-1.3B-Control" \
  --trainable_models "dit" \
  --extra_inputs "control_video,reference_image" \
  --val_interval 5 \
  --save_epochs 10 \
  --dataset AgisubDataset \