#!/bin/bash

export HF_HOME=.cache/ # moves cache to current working directory
export MODEL_PATH="THUDM/CogVideoX-5b-I2V"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Function to check if a port is available
check_port() {
  local port=$1
  # Try to bind to the port to see if it's available
  if ! (echo >/dev/tcp/127.0.0.1/$port) 2>/dev/null; then
    return 0  # Port is available
  else
    return 1  # Port is in use
  fi
}

# Find an available port with retries
MAX_ATTEMPTS=10
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
  # Use a wider range (30000-65000) to reduce collision probability
  RANDOM_PORT=$((30000 + RANDOM % 35000))
  
  if check_port $RANDOM_PORT; then
    echo "Found available port: $RANDOM_PORT (attempt $attempt)"
    break
  else
    echo "Port $RANDOM_PORT is in use, trying another... (attempt $attempt)"
    attempt=$((attempt+1))
    # Short sleep to allow for system changes
    sleep 1
  fi
done

if [ $attempt -gt $MAX_ATTEMPTS ]; then
  echo "Failed to find an available port after $MAX_ATTEMPTS attempts. Exiting."
  exit 1
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --force_type)
      FORCE_TYPE="$2"
      shift 2
      ;;
    --model_type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --num_validation_videos)
      NUM_VALIDATION_VIDEOS="$2"
      shift 2
      ;;
    --csv_path_val)
      CSV_PATH_VAL="$2"
      shift 2
      ;;
    --pretrained_controlnet_path)
      PRETRAINED_CONTROLNET_PATH="$2"
      shift 2
      ;;
    *)
      # Skip unknown arguments
      shift
      ;;
  esac
done

# replaces csv fname with images dir
IMAGE_ROOT_DIR_VAL=$(echo "$CSV_PATH_VAL" | sed 's|/[^/]*\.csv$|/images|')

# Display the values being used
echo "Using force_type:                   $FORCE_TYPE"
echo "Using num_validation_videos:        $NUM_VALIDATION_VIDEOS"
echo "Using csv_path_val:                 $CSV_PATH_VAL"
echo "Using image_root_dir_val:           $IMAGE_ROOT_DIR_VAL"
echo "Using pretrained_controlnet_path:   $PRETRAINED_CONTROLNET_PATH"
echo "Using model_type:                   $MODEL_TYPE"

# if you are not using wth 8 gus, change accelerate_config_machine_single.yaml num_processes as your gpu number
accelerate launch --config_file scripts/accelerate/accelerate_config_1_gpu.yaml \
  --main_process_port $RANDOM_PORT \
  --multi_gpu \
  src/force-prompting/train.py \
  --gradient_checkpointing \
  --csv_path_val "$CSV_PATH_VAL" \
  --image_root_dir_val "$IMAGE_ROOT_DIR_VAL" \
  --csv_path "" \
  --video_root_dir "" \
  --model_type "$MODEL_TYPE" \
  --num_inference_steps 50 \
  --num_validation_videos $NUM_VALIDATION_VIDEOS \
  --validation_steps 1000000000 \
  --controlnet_type "$FORCE_TYPE" \
  --num_train_epochs 40 \
  --tracker_name "cogvideox-controlnet" \
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
  --checkpointing_steps 250 \
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
  --pretrained_controlnet_path "$PRETRAINED_CONTROLNET_PATH" \
  --skip_training_and_only_generate_val_videos