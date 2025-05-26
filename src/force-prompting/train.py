# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import logging
import math
from pathlib import Path
from arguments import get_args

import torch
from torchvision.utils import save_image
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import diffusers
from diffusers import CogVideoXDPMScheduler
from diffusers.optimization import get_scheduler
    
from diffusers.training_utils import cast_training_params #, clear_objs_and_retain_memory,
from diffusers.utils import check_min_version, export_to_video, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card

from pipelines.controlnet_img2vid_pipeline import CogVideoXImageToVideoControlnetPipeline
from einops import rearrange

from utils.model_utils import compute_prompt_embeddings, get_optimizer, load_models, unwrap_model, clear_objs_and_retain_memory
from utils.video_utils import prepare_rotary_positional_embeddings, encode_video, tensor_to_video_ffmpeg

from data.controlnet_datasets import (
    ForcePromptingDataset_PointForce,
    ForcePromptingDataset_WindForce,
)
from data.data_utils import (
    collate_fn_ForcePromptingDataset_PointForce,
    collate_fn_ForcePromptingDataset_WindForce,
)

import datetime
import numpy as np
import cv2

if is_wandb_available():
    import wandb

from inference import (
    do_inference,
    get_object_description_point_force,
    get_baseline_prompt_point_force,
    get_baseline_prompt_wind_force
)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)

def get_dataloader_constructors(controlnet_type):
    if controlnet_type == "point_force":
        DatasetConstructor = ForcePromptingDataset_PointForce
        collate_fn = collate_fn_ForcePromptingDataset_PointForce
    elif controlnet_type == "wind_force":
        DatasetConstructor = ForcePromptingDataset_WindForce
        collate_fn = collate_fn_ForcePromptingDataset_WindForce
    else:
        raise NotImplementedError
    
    return DatasetConstructor, collate_fn

import functools
import json

def precompute_text_embeddings(
        args, tokenizer, text_encoder, device, weight_dtype, max_text_seq_length, 
        split="train", embeddings_dirname="precomputed_embeddings"
    ):
    """
    Precompute text embeddings for all prompts in the CSV and save them to disk.
    
    Returns:
        dict: Mapping from prompt text to file path containing the embedding
    """

    if split == "train":
        csv_path = args.csv_path
        video_root_dir = args.video_root_dir
    elif split == "val":
        csv_path = args.csv_path_val
        video_root_dir = args.image_root_dir_val

    dirname = os.path.dirname(csv_path)
    embeddings_dir = os.path.join(dirname, embeddings_dirname)
    os.makedirs(embeddings_dir, exist_ok=True)

    import pandas as pd
    train_df = pd.read_csv(csv_path)

    # Conditional logic for if OpenVid is there; so that we only consider text prompts which have videos...
    if "OpenVid-1M" in csv_path:
        import glob
        videos_paths = glob.glob(os.path.join(video_root_dir, '*.mp4'))
        videos_names = set([os.path.basename(x) for x in videos_paths])
        
        train_df['checked'] = train_df['video'].map(lambda x: int(x in videos_names))
        train_df = train_df[train_df['checked'] == True]
    
    # Get unique prompts from both datasets
    if split == "train":
        all_prompts = set(train_df['caption'].unique())#.union(set(val_df['caption'].unique()))
    elif split == "val":
        all_prompts = set()
        for i in range(len(train_df)):

            prompt = train_df.iloc[i]["caption"]
            all_prompts = all_prompts.union([prompt])
            
            # add force string to end of prompt if it's in our benchmark (i.e. if we have created an object description)
            if args.model_type in ["baseline_with_append_force_string_prompt", "baseline_finetune_with_append_force_string_prompt"]:
                force = train_df.iloc[i]["force"]
                angle = train_df.iloc[i]["angle"]
                file_id = train_df.iloc[i]["image"].split(".png")[0]

                if args.controlnet_type == "point_force":
                    object_description = get_object_description_point_force(file_id)
                    baseline_prompt = get_baseline_prompt_point_force(prompt, object_description, force, angle)
                elif args.controlnet_type == "wind_force":
                    baseline_prompt = get_baseline_prompt_wind_force(prompt, force, angle)
                all_prompts = all_prompts.union([baseline_prompt])

    all_prompts.add('') # for negative prompt 
    print(f"Found {len(all_prompts)} unique prompts to precompute...")

    # check if we already computed these; if we did, skip
    csv_basename = os.path.basename(csv_path).split(".csv")[0]
    if split == "train":
        embedding_map_json_path = os.path.join(embeddings_dir, f"_embedding_map_{len(all_prompts)}__{csv_basename}.json")
    elif split == "val":
        inference_type = args.model_type
        embedding_map_json_path = os.path.join(embeddings_dir, f"_embedding_map_{len(all_prompts)}_val_{csv_basename}_{inference_type}.json")

    if os.path.isfile(embedding_map_json_path):
        print("... never mind, we already computed and saved all these embeddings! Will just read the json directly.")
        with open(embedding_map_json_path, 'r') as file:
            embedding_map = json.load(file)
        return embedding_map
    
    # Create mapping from prompt to filepath
    embedding_map = {}
    
    # Process each unique prompt
    for idx, prompt in tqdm(enumerate(all_prompts)):
        # Create a unique filename based on a hash of the prompt
        import hashlib 
        prompt_hash = str(hashlib.md5(prompt.encode('utf-8')).hexdigest())
        embedding_path = os.path.join(embeddings_dir, f"embedding_{prompt_hash}.pt")
        embedding_map[prompt] = embedding_path
        
        # Skip if already computed
        if os.path.exists(embedding_path):
            print(f"Embedding for prompt {idx+1}/{len(all_prompts)} already exists, skipping")
            continue

        prompt_embeds = compute_prompt_embeddings( # [1, 226, 4096]
            tokenizer,
            text_encoder,
            [prompt],
            max_text_seq_length,
            device,
            weight_dtype,
            requires_grad=False,
        )
        
        # Save embedding to disk
        torch.save(prompt_embeds.detach().to(weight_dtype), embedding_path)
        if idx % 1000 == 0:
            print(f"Periodic printout--saved embedding {idx+1}/{len(all_prompts)}: {prompt}")
    
    # Save the mapping as JSON
    with open(embedding_map_json_path, "w") as f:
        # Convert to a dict with string keys (prompts) and string values (filepaths)
        json_map = {k: str(v) for k, v in embedding_map.items()}
        json.dump(json_map, f, indent=4)
    
    return embedding_map

# Option 1: Use a global variable for the embedding map
embedding_map_global = {}

# Function to load embeddings with LRU cache
@functools.lru_cache(maxsize=64)  # Cache most recent 64 embeddings
def load_text_embedding(prompt):
    """
    Load a precomputed text embedding from disk using the mapping.
    Uses LRU cache to keep the most recently used embeddings in memory.
    """
    embedding_path = embedding_map_global.get(prompt)
    if not embedding_path:
        raise ValueError(f"No precomputed embedding found for prompt: {prompt}")
    
    return torch.load(embedding_path, weights_only=True)

# Modified compute_prompt_embeddings function to use the precomputed embeddings
def compute_prompt_embeddings_from_cache(
    prompts, embedding_map, device, requires_grad=False
):
    """
    Get prompt embeddings from precomputed cache instead of computing them on the fly.
    """

    # Set the global embedding map
    global embedding_map_global
    embedding_map_global = embedding_map

    # Load embeddings for each prompt in the batch
    batch_embeddings = [load_text_embedding(prompt)[0] for prompt in prompts]
    
    # Stack embeddings into a single tensor
    prompt_embeds = torch.stack(batch_embeddings).to(device) # (1,226,4096)
    
    # If requires_grad, clone and set requires_grad
    if requires_grad:
        prompt_embeds = prompt_embeds.clone().detach().requires_grad_(True)
        
    return prompt_embeds


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    
    # first, update output_dir so it writes into a timestamped subdir...
    if args.pretrained_controlnet_path: # in this case, write to same directory...

        print("-"*100, "\nargs.pretrained_controlnet_path = True, so we're overwriting output dir to that directory.")
        print("args.pretrained_controlnet_path = ", args.pretrained_controlnet_path, "\n", "-"*100)
        
        # Extract step number from filename (e.g., step-2100-checkpoint.pt → 2100)
        # checkpoint_filename = os.path.basename(args.pretrained_controlnet_path)
        global_step = int(os.path.basename(args.pretrained_controlnet_path).split(".")[0].split("-")[1]) - 1
        
        # Set the output directory to the same directory as the checkpoint
        args.output_dir = os.path.dirname(args.pretrained_controlnet_path)
        
        # Define paths to optimizer and scheduler states
        # optimizer_path = os.path.join(args.output_dir, f"step-{global_step}-optimizer.pt")
        # scheduler_path = os.path.join(args.output_dir, f"step-{global_step}-scheduler.pt")
        
        # print(f"Will attempt to load optimizer state from: {optimizer_path}")
        # print(f"Will attempt to load scheduler state from: {scheduler_path}")

    else:
        print("TO DO: only have the first process create the datetime string, and broadcast it to the other processes...")
        datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output_dir = os.path.join(args.output_dir, datetime_string)
        global_step = 0

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        
        if args.output_dir is not None and not args.skip_training_and_only_generate_val_videos:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

        # Later in your code where you set up the output directory:
        if args.launch_script_path and os.path.exists(args.launch_script_path):
            script_filename = os.path.basename(args.launch_script_path)
            output_script_path = os.path.join(args.output_dir, script_filename)
            shutil.copy2(args.launch_script_path, output_script_path)
            print(f"Copied launch script to {output_script_path}")


    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin: # "False"
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.float16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16


    # Load models
    models = load_models(args)
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    transformer = models["transformer"]
    vae = models["vae"]
    controlnet = models["controlnet"]
    scheduler = models["scheduler"]

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config


    # We only train the additional adapter controlnet layers
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    controlnet.requires_grad_(True)

    # Precompute embeddings before training starts
    if accelerator.is_main_process and not args.skip_training_and_only_generate_val_videos:
        print("Precomputing text embeddings...")
        embedding_map = precompute_text_embeddings(
            args, tokenizer, text_encoder, accelerator.device, weight_dtype, model_config.max_text_seq_length
        )
        embedding_map_save_path_temp = os.path.join(args.output_dir, "embedding_map_temp.pt")
        torch.save(embedding_map, embedding_map_save_path_temp)
    else:
        embedding_map = None

    if not args.skip_training_and_only_generate_val_videos:
        # Make sure all processes wait until the main process has saved the file
        accelerator.wait_for_everyone()
        # Now all processes can load the file
        save_path = os.path.join(args.output_dir, "embedding_map_temp.pt")
        embedding_map = torch.load(save_path)

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    try:
        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing()
            controlnet.enable_gradient_checkpointing()
    except:
        transformer.enable_gradient_checkpointing()
        # controlnet.enable_gradient_checkpointing() doesn't work so we need to do it manually..
        def _set_gradient_checkpointing(model): 
            # Enable checkpointing for all applicable modules
            for module in model.modules():
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = True
        _set_gradient_checkpointing(controlnet)



    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters into fp32
        cast_training_params([controlnet], dtype=torch.float32)

    trainable_parameters = list(filter(lambda p: p.requires_grad, controlnet.parameters()))

    # Optimization parameters
    trainable_parameters_with_lr = {"params": trainable_parameters, "lr": args.learning_rate}
    params_to_optimize = [trainable_parameters_with_lr]

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(args, params_to_optimize, logger, use_deepspeed=use_deepspeed_optimizer)

    # Dataset and DataLoader
    DatasetConstructor, collate_fn = get_dataloader_constructors(args.controlnet_type)

    if args.skip_training_and_only_generate_val_videos:
        val_dataset = DatasetConstructor(
            video_root_dir=args.image_root_dir_val,
            csv_path=args.csv_path_val,
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            controlnet_type=args.controlnet_type,
            is_validation_dataset=True
        )
        # need to overwrite to values in the training dataset
        val_dataset.min_force = 0.0
        val_dataset.max_force = 1.0

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=1,
        )
    else:

        train_dataset = DatasetConstructor(
            video_root_dir=args.video_root_dir,
            csv_path=args.csv_path,
            image_size=(args.height, args.width), 
            stride=(args.stride_min, args.stride_max),
            sample_n_frames=args.max_num_frames,
            controlnet_type=args.controlnet_type,

        )


        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.dataloader_num_workers,
        )


    if args.skip_training_and_only_generate_val_videos:

        embedding_map = precompute_text_embeddings(
            args, tokenizer, text_encoder, accelerator.device, weight_dtype, model_config.max_text_seq_length, split="val"
        )

        del models["text_encoder"]
        del text_encoder
        torch.cuda.empty_cache()
        text_encoder = None

        logger.info("***** Running validation *****")
        do_inference(
            accelerator,
            transformer,
            text_encoder,
            vae,
            controlnet,
            scheduler,
            weight_dtype,
            val_dataloader,
            args.controlnet_type,
            global_step=global_step,
            embedding_map=embedding_map,
            model_type=args.model_type,
            args=args
        )
        logger.info("***** Done running validation *****")
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return None
    
    if args.validation_steps > 1000000:
        # Unload text encoder to free up memory; should find a way to do this BEFORE the do_validation perhaps?
        del models["text_encoder"]
        del text_encoder
        torch.cuda.empty_cache()
        text_encoder = None

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    # Load optimizer and scheduler states if resuming from checkpoint
    if args.pretrained_controlnet_path and global_step > 0:
        # Construct paths to the optimizer and scheduler state files
        optimizer_path = os.path.join(args.output_dir, f"step-{global_step+1}-optimizer.pt")
        scheduler_path = os.path.join(args.output_dir, f"step-{global_step+1}-scheduler.pt")
        
        # Load optimizer state if it exists
        if os.path.exists(optimizer_path):
            try:
                optimizer_state = torch.load(optimizer_path, map_location="cpu")
                optimizer.load_state_dict(optimizer_state)
                print(f"Successfully loaded optimizer state from {optimizer_path}")
            except Exception as e:
                print(f"Failed to load optimizer state: {e}")
                # If loading fails, we'll continue with freshly initialized optimizer
        else:
            print(f"No optimizer state found at {optimizer_path}, using fresh optimizer")
        
        # Create the scheduler first
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
        
        # Then try to load its state if exists
        if os.path.exists(scheduler_path):
            try:
                scheduler_state = torch.load(scheduler_path, map_location="cpu")
                lr_scheduler.load_state_dict(scheduler_state)
                print(f"Successfully loaded scheduler state from {scheduler_path}")
            except Exception as e:
                print(f"Failed to load scheduler state: {e}")
                # If loading fails, we'll step the scheduler to match global step
                print(f"Manually advancing scheduler to step {global_step}")
                for _ in range(global_step):
                    lr_scheduler.step()
        else:
            print(f"No scheduler state found at {scheduler_path}, advancing scheduler to step {global_step}")
            for _ in range(global_step):
                lr_scheduler.step()
    else:
        # Create new scheduler if not resuming
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    # Prepare everything with our `accelerator`. wraps them in accelerate classes
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-controlnet"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    first_epoch = 0
    initial_global_step = global_step

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)

    for epoch in range(first_epoch, args.num_train_epochs):
        controlnet.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [controlnet]

            with accelerator.accumulate(models_to_accumulate):
                # [1, 49, 3, 480, 720] --> [1, 13, 16, 50, 90] = [B, F, C, H, W]
                model_input = encode_video(vae, accelerator, batch["videos"]).to(dtype=weight_dtype)  
                # Q: Do we actually need to encode these controlnet frames? They dont have the right shape acc. their name...
                # A: no! We want to use custom encoding logic (that is part of the controlnet)
                controlnet_encoded_frames = batch["controlnet_videos"] # [1, 49, 3, 480, 720]
                # print(controlnet_encoded_frames.min(), controlnet_encoded_frames.max())
                # tensor_to_video_ffmpeg(torch.clip(0.5 + 0.5*batch["videos"][0] + controlnet_encoded_frames[0], max=1.0), f"output/temp/controlnet_frames_{step:03d}.mp4")
                prompts = batch["prompts"] # List[Str]
                
                # encode prompts
                prompt_embeds = compute_prompt_embeddings_from_cache(
                    prompts, embedding_map, accelerator.device, requires_grad=False
                )
                # prompt_embeds = compute_prompt_embeddings( # [1, 226, 4096]
                #     tokenizer,
                #     text_encoder,
                #     prompts,
                #     model_config.max_text_seq_length,
                #     accelerator.device,
                #     weight_dtype,
                #     requires_grad=False,
                # )

                batch_size, num_frames, num_channels, height, width = model_input.shape # (1, 13, 16, 60, 90)

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (batch_size,), device=model_input.device
                )
                timesteps = timesteps.long()
        
                # Prepare rotary embeds
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=args.height,
                        width=args.width,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=vae_scale_factor_spatial,
                        patch_size=model_config.patch_size,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                    )
                    if model_config.use_rotary_positional_embeddings # True
                    else None
                )

                # We add image conditioning following finetune/models/cogvideox_i2v/lora_trainer.py in original CogVideo repo
                images = batch["first_frames"] # (1, 3, 480, 720) = [B,C,H,W]
                # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
                images = images.unsqueeze(2) # (1, 3, 1, 480, 720) = [B,C,F,H,W]
                image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=accelerator.device)
                image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
                noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
                image_latent_dist = vae.encode(noisy_images.to(dtype=vae.dtype)).latent_dist # diffusers.models.autoencoders.vae.DiagonalGaussianDistribution
                image_latents = image_latent_dist.sample() * vae.config.scaling_factor # (1, 16, 1, 60, 90)
                image_latents = rearrange(image_latents, 'b c f h w -> b f c h w') # (1, 1, 16, 60, 90)
                # Padding image_latents to the same frame number as model_input (i.e. the video latent)
                padding_shape = (model_input.shape[0], model_input.shape[1] - 1, *model_input.shape[2:])
                latent_padding = image_latents.new_zeros(padding_shape) # (1, 12, 16, 60, 90)
                image_latents = torch.cat([image_latents, latent_padding], dim=1) # (1, 13, 16, 60, 90)

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # (1, 13, 16, 60, 90), (1, 13, 16, 60, 90), (1) --> (1, 13, 16, 60, 90)
                # Sample noise that will be added to the latents
                noise = torch.randn_like(model_input) # [1, 13, 16, 60, 90]
                noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)

                # Concatenate noisy_model_input and image_latents; the link above said do it in the channel dimension,
                # but you cant use the pretrained weights if you do that, so I think you should do it in the frame dimension
                noisy_model_input_and_image_latents = torch.cat([noisy_model_input, image_latents], dim=2) # (1, 26, 32, 60, 90)
                # NOTE: seems very inefficient to have these extra dimensions from all that padding... ask

                controlnet_states = controlnet(
                    hidden_states=noisy_model_input_and_image_latents, # (1, 13, 32, 60, 90)
                    encoder_hidden_states=prompt_embeds, # (1, 226, 4096)
                    image_rotary_emb=image_rotary_emb, # tuple of len 2, each entry of shape (17550, 64)
                    controlnet_states=controlnet_encoded_frames, # (1, 49, 3, 480, 720); these aren't actually encoded?
                    timestep=timesteps, # (1,)
                    return_dict=False,
                )[0]
                if isinstance(controlnet_states, (tuple, list)):
                    # controlnet_states[i].shape = (1, 17550, 1920) for i \in {0, ..., 7}. one for every transformer layer!
                    controlnet_states = [x.to(dtype=weight_dtype) for x in controlnet_states]
                else:
                    controlnet_states = controlnet_states.to(dtype=weight_dtype)
                # Predict the noise residual
                model_output = transformer(
                    hidden_states=noisy_model_input_and_image_latents, # (1, 13, 32, 60, 90)
                    encoder_hidden_states=prompt_embeds, # (1, 226, 4096)
                    timestep=timesteps, # (1,)
                    image_rotary_emb=image_rotary_emb, # None
                    controlnet_states=controlnet_states, # controlnet_states[i].shape = (1, 17550, 1920) for i \in {0, ..., 7}. one for every transformer layer!
                    controlnet_weights=args.controlnet_weights, # 0.5
                    return_dict=False,
                )[0] # (1, 13, 16, 60, 90)

                # (1, 13, 16, 60, 90), (1, 13, 16, 60, 90), (1,) --> (1, 13, 16, 60, 90)
                model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)

                alphas_cumprod = scheduler.alphas_cumprod[timesteps] # (1,)
                weights = 1 / (1 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = model_input

                loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1) # (1,)
                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0 and global_step > initial_global_step+10:
                        save_path = os.path.join(args.output_dir, f"step-{global_step}-checkpoint.pt")
                        torch.save({'state_dict': unwrap_model(accelerator, controlnet).state_dict()}, save_path)

                        # Save optimizer state
                        optimizer_path = os.path.join(args.output_dir, f"step-{global_step}-optimizer.pt")
                        torch.save(optimizer.state_dict(), optimizer_path)
                        
                        # Save scheduler state
                        scheduler_path = os.path.join(args.output_dir, f"step-{global_step}-scheduler.pt")
                        torch.save(lr_scheduler.state_dict(), scheduler_path)

                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    main(args)