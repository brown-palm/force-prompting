#!/bin/bash

export HF_HOME=.cache/ # moves cache to current working directory
export MODEL_PATH="THUDM/CogVideoX-5b-I2V"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3 # e.g. 0 or 0,1,2,3
export TOKENIZERS_PARALLELISM=false
export LAUNCH_SCRIPT_PATH="$(readlink -f "$0")"

# Default values
RESUME_FROM_CHECKPOINT=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --force_type)
      FORCE_TYPE="$2"
      shift 2
      ;;
    --video_root_dir)
      VIDEO_ROOT_DIR="$2"
      shift 2
      ;;
    --csv_path)
      CSV_PATH="$2"
      shift 2
      ;;
    --pretrained_controlnet_path)
      PRETRAINED_CONTROLNET_PATH="$2"
      RESUME_FROM_CHECKPOINT=true
      shift 2
      ;;
    *)
      # Skip unknown arguments
      shift
      ;;
  esac
done

# Display the values being used
echo "Using force_type:     $FORCE_TYPE"
echo "Using video_root_dir: $VIDEO_ROOT_DIR"
echo "Using csv_path:       $CSV_PATH"

# Construct the command
CMD="accelerate launch --config_file scripts/accelerate/accelerate_config_4_gpu.yaml --multi_gpu \
  src/force-prompting/train.py \
  --gradient_checkpointing \
  --csv_path_val \"\" \
  --image_root_dir_val \"\" \
  --num_inference_steps 50 \
  --num_validation_videos 4 \
  --validation_steps 1000000000 \
  --video_root_dir \"$VIDEO_ROOT_DIR\" \
  --csv_path \"$CSV_PATH\" \
  --controlnet_type \"$FORCE_TYPE\" \
  --num_train_epochs 40 \
  --tracker_name \"cogvideox-controlnet\" \
  --pretrained_model_name_or_path $MODEL_PATH \
  --enable_tiling \
  --enable_slicing \
  --seed 42 \
  --mixed_precision bf16 \
  --output_dir \"output/$FORCE_TYPE\" \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --stride_min 1 \
  --stride_max 3 \
  --controlnet_transformer_num_layers 6 \
  --controlnet_input_channels 3 \
  --downscale_coef 8 \
  --controlnet_weights 1.0 \
  --init_from_transformer \
  --train_batch_size 1 \
  --dataloader_num_workers 1 \
  --checkpointing_steps 500 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 250 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --launch_script_path \"$LAUNCH_SCRIPT_PATH\" \
  --report_to wandb"

# Add pretrained_controlnet_path if provided
if $RESUME_FROM_CHECKPOINT; then
  echo "Resuming from checkpoint: $PRETRAINED_CONTROLNET_PATH"
  CMD="$CMD \
  --pretrained_controlnet_path \"$PRETRAINED_CONTROLNET_PATH\""
else
  echo "Training from scratch"
fi

# Execute the command
eval $CMD