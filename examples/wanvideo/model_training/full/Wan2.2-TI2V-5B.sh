CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 accelerate launch \
  --config_file /mnt/workspace/zsq/DiffSynth-Studio/examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  /mnt/workspace/zsq/DiffSynth-Studio/examples/wanvideo/model_training/train.py \
  --num_frames 9 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_full" \
  --trainable_models "dit" \
  --extra_inputs "input_image"