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
import json

if is_wandb_available():
    import wandb

def get_baseline_prompt_point_force(prompt, object_description, force, angle):

    if 0 <= force <= 0.25:
        force_str = "is moved not very forcefully"
    elif 0.25 <= force <= 0.75:
        force_str = "is moved forcefully"
    elif 0.75 <= force <= 1.0:
        force_str = "is moved very forcefully"


    if (0 <= angle <= 22.5) or (337.5 <= angle <= 360):
        angle_str = "to the right"
    elif 22.5 <= angle <= 67.5:
        angle_str = "upwards and to the right"
    elif 67.5 <= angle <= 112.5:
        angle_str = "upwards"
    elif 112.5 <= angle <= 157.5:
        angle_str = "upwards and to the left"
    elif 157.5 <= angle <= 202.5:
        angle_str = "to the left"
    elif 202.5 <= angle <= 247.5:
        angle_str = "downwards and to the left"
    elif 247.5 <= angle <= 292.5:
        angle_str = "downwards"
    elif 292.5 <= angle <= 337.5:
        angle_str = "downwards and to the right"
    else:
        raise ValueError

    force_str = f"The {object_description} {force_str}, {angle_str}"

    prompt_without_end_punctuation = prompt
    while prompt_without_end_punctuation[-1].lower() not in "abcdefghijklmnopqrstuvwxyz":
        prompt_without_end_punctuation = prompt_without_end_punctuation[:-1]

    baseline_prompt = f"{prompt_without_end_punctuation}. {force_str}"

    return baseline_prompt

def get_baseline_prompt_wind_force(prompt, wind_speed, wind_angle, normalization=15000):

    if 0 <= wind_speed <= 2000/normalization:
        wind_speed_str = "The wind is very soft"
    elif 2000/normalization < wind_speed <= 6000/normalization:
        wind_speed_str = "The wind is soft"
    elif 6000/normalization < wind_speed <= 10000/normalization:
        wind_speed_str = "The wind is medium strength"
    elif 10000/normalization < wind_speed:
        wind_speed_str = "The wind is very strong"
    else:
        raise ValueError

    if (0 <= wind_angle <= 30) or (330 <= wind_angle <= 360):
        wind_angle_str = "blowing to the right"
    elif 30 <= wind_angle <= 70:
        wind_angle_str = "blowing upwards and to the right"
    elif 70 <= wind_angle <= 110:
        wind_angle_str = "blowing upwards"
    elif 110 <= wind_angle <= 150:
        wind_angle_str = "blowing upwards and to the left"
    elif 150 <= wind_angle <= 210:
        wind_angle_str = "blowing to the left"
    elif 210 <= wind_angle <= 250:
        wind_angle_str = "blowing downwards and to the left"
    elif 250 <= wind_angle <= 290:
        wind_angle_str = "blowing downwards"
    elif 290 <= wind_angle <= 330:
        wind_angle_str = "blowing downwards and to the right"
    else:
        raise ValueError

    wind_str = f"{wind_speed_str}, {wind_angle_str}"

    prompt_without_end_punctuation = prompt
    while prompt_without_end_punctuation[-1].lower() not in "abcdefghijklmnopqrstuvwxyz":
        prompt_without_end_punctuation = prompt_without_end_punctuation[:-1]

    baseline_prompt = f"{prompt_without_end_punctuation}. {wind_str}"

    return baseline_prompt

def add_aesthetic_point_force_prompt_to_video(video, force, angle, x_pos, y_pos, circle_radius=20, num_frames_with_signal=1):
    """
    Annotate the first frame of a video with a white circle and directional yellow arrow.
    
    Parameters:
    -----------
    video : numpy.ndarray
        Video array with shape (num_frames, height, width, channels), values in [0,1]
    force : float
        Value in [0,1] that determines the length of the arrow
    angle : float
        Value in [0,360] that determines the direction of the arrow
    x_pos : float
        Horizontal position in [0,1] (will be scaled to pixel coordinates)
    y_pos : float
        Vertical position in [0,1] (will be scaled to pixel coordinates)
    
    Returns:
    --------
    numpy.ndarray
        Modified video with annotations on the first frame
    """
    # Create a copy of the video to avoid modifying the original
    result_video = video.copy()
    
    # Get the dimensions of the video
    num_frames, height, width, channels = video.shape
    
    # Convert the position from [0,1] range to pixel coordinates
    center_x = int(x_pos * width)
    center_y = int(y_pos * height)
    
    # Convert angle from degrees to radians
    angle_rad = math.radians(angle)
    
    # Calculate the arrow endpoint
    arrow_length = 10 + 90 * force # min force in dataset, corresponidng to 0, should have some positive length...
    end_x = int(center_x + arrow_length * math.cos(angle_rad))
    end_y = int(center_y - arrow_length * math.sin(angle_rad))

    for i in range(num_frames_with_signal):
        # Convert the first frame to uint8 format (0-255) for OpenCV
        this_frame = (result_video[i] * 255).astype(np.uint8)
        
        # Draw a white circle with radius 10 pixels and thickness 2 pixels
        cv2.circle(this_frame, (center_x, center_y), circle_radius, (255, 255, 255), 2)
        
        # Draw a yellow arrow
        cv2.arrowedLine(this_frame, (center_x, center_y), (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)
        
        # Convert the frame back to [0,1] range
    
        result_video[i] = this_frame / 255.0
    
    return result_video

# Update max_arrow_length properly to 90 to account for forward distance
def add_aesthetic_wind_force_prompt_to_video(
    video,
    force,
    angle,
    num_frames_with_signal=1,
    base_periods=1,
    periods_per_0_1_force=1,
    wave_amplitude=2,
    extra_straight_length=20,
    arrowhead_length=7,
    forward_distance=6
):
    result_video = video.copy()
    num_frames, height, width, channels = video.shape

    arrowhead_base = int(arrowhead_length * (2 / math.sqrt(3)))

    min_arrow_length = 30
    max_arrow_length = 90  # final correct value

    arrow_length = min_arrow_length + force * (max_arrow_length - min_arrow_length)
    periods = base_periods + int(force * 10) * periods_per_0_1_force

    angle_rad = math.radians(angle)
    dir_x = math.cos(angle_rad)
    dir_y = -math.sin(angle_rad)
    perp_x = -dir_y
    perp_y = dir_x

    base_x = width - 100
    base_y = 100

    for i in range(min(num_frames_with_signal, num_frames)):
        frame = (result_video[i] * 255).astype(np.uint8)

        for j in range(3):
            offset = (j - 1) * 20
            start_x = base_x + offset * perp_x
            start_y = base_y + offset * perp_y

            points = []
            num_points = 100
            squiggly_part_length = arrow_length - extra_straight_length
            squiggly_end_t = squiggly_part_length / arrow_length

            for k in range(num_points):
                t = k / (num_points - 1)
                if t < squiggly_end_t:
                    main_x = start_x + dir_x * t * arrow_length
                    main_y = start_y + dir_y * t * arrow_length
                    squiggle = math.sin(t * periods * 2 * math.pi) * wave_amplitude
                    squiggle_x = main_x + perp_x * squiggle
                    squiggle_y = main_y + perp_y * squiggle
                else:
                    straight_progress = (t - squiggly_end_t) / (1 - squiggly_end_t)
                    main_x = start_x + dir_x * (squiggly_part_length + straight_progress * extra_straight_length)
                    main_y = start_y + dir_y * (squiggly_part_length + straight_progress * extra_straight_length)
                    squiggle_x = main_x
                    squiggle_y = main_y

                points.append((int(squiggle_x), int(squiggle_y)))

            for p in range(len(points) - 1):
                cv2.line(frame, points[p], points[p + 1], (0, 255, 255), 2)

            tip = points[-1]
            tip_forward_x = tip[0] + forward_distance * dir_x
            tip_forward_y = tip[1] + forward_distance * dir_y
            tip_point = (int(tip_forward_x), int(tip_forward_y))

            base_center_x = tip[0] - arrowhead_length * dir_x
            base_center_y = tip[1] - arrowhead_length * dir_y

            left_base_x = int(base_center_x + (arrowhead_base / 2) * -dir_y)
            left_base_y = int(base_center_y + (arrowhead_base / 2) * dir_x)

            right_base_x = int(base_center_x - (arrowhead_base / 2) * -dir_y)
            right_base_y = int(base_center_y - (arrowhead_base / 2) * dir_x)

            cv2.line(frame, (left_base_x, left_base_y), tip_point, (0, 255, 255), 2)
            cv2.line(frame, (right_base_x, right_base_y), tip_point, (0, 255, 255), 2)

        result_video[i] = frame / 255.0

    return result_video

def get_object_description_point_force(file_id):

    file_id_to_object_description = {
        "_apple1" : "apple",
        "_apple2" : "apple",
        "_apple3" : "apple",
        "_apple4" : "apple",
        "_balloon3" : "hot air balloon",
        "_balloon4" : "balloon",
        "_blueberrybush1" : "blueberry",
        "_blueberrybush2" : "blueberry",
        "_blueberrybush3" : "blueberry",
        "_blueberrybush4" : "blueberry",
        "_dandelion1" : "dandelion",
        "_dandelion3" : "dandelion",
        "_dandelion4" : "dandelion",
        "_ivy1" : "ivy",
        "_ornament1" : "bear ornament",
        "_ornament2" : "horse ornament",
        "_ornament4" : "star ornament",
        "_rose2" : "rose",
        "_rose3" : "rose",
        "_rose4" : "rose",
        "_rose5" : "rose",
        "_sunflower2" : "sunflower",
        "_sunflower3" : "sunflower",
        "_swing3" : "swinging chair",
        "_toycar1" : "toy car",
        "_toycar2" : "toy car",
        "_toycar3" : "toy car",
        "_toycar4" : "toy bus",
        "_toytrainontrack1" : "toy train",
        "_toytrainontrack2" : "toy train",
        "_toytrainontrack3" : "toy train",
        "_toytrainontrack5" : "toy train",
        "_toytrainontrack6" : "toy train",
        "_toytrainontrack7" : "toy train",
        "_windmill2" : "top of the windmill",
    }

    assert file_id in file_id_to_object_description
    return file_id_to_object_description[file_id]

def do_inference(
    accelerator,
    transformer,
    text_encoder,
    vae,
    controlnet,
    scheduler,
    weight_dtype,
    val_dataloader,
    controlnet_type,
    epoch=0,
    global_step=0,
    embedding_map={},
    model_type="controlnet_with_force_control_signal",
    args=None
):

    # Create pipeline
    pipe = CogVideoXImageToVideoControlnetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=unwrap_model(accelerator, transformer),
        text_encoder=unwrap_model(accelerator, text_encoder),
        vae=unwrap_model(accelerator, vae),
        controlnet=unwrap_model(accelerator, controlnet),
        scheduler=scheduler,
        torch_dtype=weight_dtype,
        cache_dir=".cache",
        local_files_only=False,
    )

    # for validation_prompt, validation_video in zip(validation_prompts, validation_videos):
    print(f"Beginning val with {len(val_dataloader)} batches...")
    for _, val_batch in enumerate(val_dataloader):

        # BASELINE: controlnet weights = 0, updated text prompt
        # PHYSICS CONTROL: controlnet weights = 1, original text prompt
        file_id = val_batch["file_ids"][0]

        if args.controlnet_type == "point_force":

            # CASE 1: point force

            force = val_batch["force"][0]
            angle = val_batch["angle"][0]
            x_pos = val_batch["x_pos"][0]
            y_pos = val_batch["y_pos"][0]

            fname_str = f"file_id_{file_id}__xpos_{x_pos:.2f}__ypos_{y_pos:.2f}__angle_{angle:.2f}__force_{force:.3f}"

            prompt = val_batch["prompts"][0]
            if model_type in ["baseline_with_append_force_string_prompt", "baseline_finetune_with_append_force_string_prompt"]:
                object_description = get_object_description_point_force(file_id)
                prompt = get_baseline_prompt_point_force(prompt, object_description, force, angle)

            INFERENCE_CONFIGS_FOR_MODE = {
                "baseline_with_original_prompt" : {
                    "controlnet_weights" : 0.0, # completely ignore controlnet signal!
                    "prompt" : prompt,
                    "fname_image_condition" : f"{fname_str}___image_condition.png",
                    "fname_base_generated_video" : f"{fname_str}",
                    "fname_text_prompt" : f"{fname_str}_baseline___prompt.json",
                    "output_dir" : os.path.join(os.path.dirname(os.path.dirname(args.image_root_dir_val)), "_videos_baseline_with_original_prompt")
                },
                "baseline_with_append_force_string_prompt" :  {
                    "controlnet_weights" : 0.0, # completely ignore controlnet signal!
                    "prompt" : prompt,
                    "fname_image_condition" : f"{fname_str}___image_condition.png",
                    "fname_base_generated_video" : f"{fname_str}_baseline",
                    "fname_text_prompt" : f"{fname_str}_baseline___prompt.json",
                    "output_dir" : os.path.join(os.path.dirname(os.path.dirname(args.image_root_dir_val)), "_videos_baseline_with_append_force_string_prompt")
                },
                "baseline_finetune_with_append_force_string_prompt" :  {
                    "controlnet_weights" : 1.0, # use the controlnet signal!
                    "prompt" : prompt,
                    "fname_image_condition" : f"{fname_str}___image_condition.png",
                    "fname_base_generated_video" : f"{fname_str}_baseline_finetune",
                    "fname_text_prompt" : f"{fname_str}_baseline_finetune___prompt.json",
                    "output_dir" : os.path.join(args.output_dir, os.path.basename(args.pretrained_controlnet_path).split(".pt")[0])
                },
                "controlnet_with_force_control_signal" : {
                    "controlnet_weights" : args.controlnet_weights,
                    "prompt" : prompt,
                    "fname_image_condition" : f"step-{global_step+1}__{fname_str}___image_condition.png",
                    "fname_base_generated_video" : f"step-{global_step+1}__{fname_str}",
                    "fname_text_prompt" : f"step-{global_step+1}__{fname_str}___prompt.json",
                    "fname_control_signal" : f"step-{global_step+1}__{fname_str}___control_signal.mp4",
                    "output_dir" : os.path.join(args.output_dir, os.path.basename(args.pretrained_controlnet_path).split(".pt")[0])
                },
            }

        elif args.controlnet_type == "wind_force": # wind force
            if "force" in val_batch and "angle" in val_batch:
                assert len(val_batch["force"]) == len(val_batch["angle"]) == 1
                force = val_batch["force"][0]
                angle = val_batch["angle"][0]
            else:
                force = -1.0
                angle = -1.0

            fname_str = f"file_id_{file_id}__force_{force:.2f}__angle_{angle:.1f}"

            prompt = val_batch["prompts"][0]
            if model_type in ["baseline_with_append_force_string_prompt", "baseline_finetune_with_append_force_string_prompt"]:
                prompt = get_baseline_prompt_wind_force(prompt, force, angle)

            INFERENCE_CONFIGS_FOR_MODE = {
                "baseline_with_original_prompt" : {
                    "controlnet_weights" : 0.0, # completely ignore controlnet signal!
                    "prompt" : prompt,
                    "fname_image_condition" : f"{fname_str}___image_condition.png",
                    "fname_base_generated_video" : f"{fname_str}",
                    "fname_text_prompt" : f"{fname_str}_baseline___prompt.json",
                    "output_dir" : os.path.join(os.path.dirname(os.path.dirname(args.image_root_dir_val)), "videos_baseline_with_original_prompt")
                },
                "baseline_with_append_force_string_prompt" :  {
                    "controlnet_weights" : 0.0, # completely ignore controlnet signal!
                    "prompt" : prompt,
                    "fname_image_condition" : f"{fname_str}___image_condition.png",
                    "fname_base_generated_video" : f"{fname_str}_baseline",
                    "fname_text_prompt" : f"{fname_str}_baseline___prompt.json",
                    "output_dir" : os.path.join(os.path.dirname(os.path.dirname(args.image_root_dir_val)), f"_videos_baseline_with_append_force_string_prompts")
                },
                "baseline_finetune_with_append_force_string_prompt" :  {
                    "controlnet_weights" : 1.0, # use the controlnet signal!
                    "prompt" : prompt,
                    "fname_image_condition" : f"{fname_str}___image_condition.png",
                    "fname_base_generated_video" : f"{fname_str}_baseline_finetune",
                    "fname_text_prompt" : f"{fname_str}_baseline_finetune___prompt.json",
                    "output_dir" : os.path.join(args.output_dir, os.path.basename(args.pretrained_controlnet_path).split(".pt")[0])
                },
                "controlnet_with_force_control_signal" : {
                    "controlnet_weights" : args.controlnet_weights,
                    "prompt" : prompt,
                    "fname_image_condition" : f"step-{global_step+1}__{fname_str}___image_condition.png",
                    "fname_base_generated_video" : f"step-{global_step+1}__{fname_str}",
                    "fname_text_prompt" : f"step-{global_step+1}__{fname_str}___prompt.json",
                    "fname_control_signal" : f"step-{global_step+1}__{fname_str}___control_signal.mp4",
                    "output_dir" : os.path.join(args.output_dir, os.path.basename(args.pretrained_controlnet_path).split(".pt")[0])
                },
            }

        for MODE in [model_type]:

            output_dir = INFERENCE_CONFIGS_FOR_MODE[MODE]["output_dir"]
            os.makedirs(output_dir, exist_ok=True)

            controlnet_weights = INFERENCE_CONFIGS_FOR_MODE[MODE]["controlnet_weights"]
            prompt = INFERENCE_CONFIGS_FOR_MODE[MODE]["prompt"]
            fname_base_generated_video = INFERENCE_CONFIGS_FOR_MODE[MODE]["fname_base_generated_video"]
            fname_text_prompt = INFERENCE_CONFIGS_FOR_MODE[MODE]["fname_text_prompt"]
            
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

            # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
            scheduler_args = {}

            if "variance_type" in pipe.scheduler.config:
                variance_type = pipe.scheduler.config.variance_type

                if variance_type in ["learned", "learned_range"]:
                    variance_type = "fixed_small"

                scheduler_args["variance_type"] = variance_type

            pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
            pipe = pipe.to(accelerator.device)

            pipeline_args = {
                "prompt": prompt, # str
                "image": val_batch["first_frames"].to(accelerator.device), # (1, 3, 480, 720)
                "controlnet_frames": val_batch["controlnet_videos"].to(accelerator.device), # (1, 49, 3, 480, 720)
                "guidance_scale": args.guidance_scale,
                "use_dynamic_cfg": args.use_dynamic_cfg,
                "height": args.height,
                "width": args.width,
                "num_frames": args.max_num_frames,
                "num_inference_steps": args.num_inference_steps,
                "controlnet_weights": controlnet_weights,
                "generator": generator,
                "output_type": "np"
            }
            if len(embedding_map) > 0: # in case we precomputed embeddings
                del[pipeline_args["prompt"]]
                pipeline_args["prompt_embeds"] = torch.load(embedding_map[prompt])
                pipeline_args["negative_prompt_embeds"] = torch.load(embedding_map[''])



            if MODE == "controlnet_with_force_control_signal":
                # save the controlnet video
                filename_control_signal = os.path.join(
                    output_dir, 
                    INFERENCE_CONFIGS_FOR_MODE[MODE]["fname_control_signal"]
                )
                control_signal_video = val_batch["controlnet_videos"] # (1, 49, 3, 480, 720), torch.float32 from [-1,1] mostly -1
                control_signal_video = rearrange(control_signal_video, 'b f c h w -> (b f) h w c') # (49, 480, 720, 3)
                export_to_video(control_signal_video.numpy(), filename_control_signal, fps=8)

            # save the conditioning image as well...
            filename_image_condition = os.path.join(
                output_dir, 
                INFERENCE_CONFIGS_FOR_MODE[MODE]["fname_image_condition"]
            )

            print(
                f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {prompt}\n\
                fname_string = {fname_str}."
            )

            image_condition = val_batch["first_frames"][0] # (3, 480, 720)
            save_image((image_condition+1)/2, filename_image_condition)

            # save the text prompt as well
            text_prompt = {"controlnet_weights" : controlnet_weights, "prompt" : prompt}
            text_prompt_save_path = os.path.join(output_dir, fname_text_prompt)
            with open(text_prompt_save_path, 'w') as f:
                json.dump(text_prompt, f, indent=4)

            videos = []
            for i in range(args.num_validation_videos):

                filename_generated_video = os.path.join(
                    output_dir, 
                    f"{fname_base_generated_video}__video_{i}.mp4"
                )

                # generate the video
                video = pipe(**pipeline_args).frames[0] # (49, 480, 720, 3)
                videos.append(video)

                # save the generated video
                export_to_video(video, filename_generated_video, fps=8)


                if MODE == "controlnet_with_force_control_signal":
                    # visualize video and control signal prompt in same video
                    filename_generated_video_with_force_prompt = os.path.join(
                        output_dir, 
                        f"{fname_base_generated_video}___video_{i}_with_control_signal.mp4"
                    )
                    video_with_force_prompt = np.clip(video + control_signal_video.numpy(), 0, 1.0)
                    export_to_video(
                        video_with_force_prompt, filename_generated_video_with_force_prompt, fps=8
                    )

                # visualize video and a pretty version of the control signal prompt in same video
                filename_generated_video_with_force_prompt_aesthetic = os.path.join(
                    output_dir, 
                    f"{fname_base_generated_video}___video_{i}_with_pretty_force_prompt.mp4"
                )
                min_force = val_dataloader.dataset.min_force
                max_force = val_dataloader.dataset.max_force
                normalized_force = (val_batch["force"][0] - min_force) / (max_force - min_force)

                if args.controlnet_type == "point_force":
                    video_with_force_prompt_aesthetic = add_aesthetic_point_force_prompt_to_video(
                        video, normalized_force, angle, x_pos, 1 - y_pos, num_frames_with_signal=8
                    )
                elif args.controlnet_type == "wind_force":
                    video_with_force_prompt_aesthetic = add_aesthetic_wind_force_prompt_to_video(
                        video, normalized_force, angle, num_frames_with_signal=49
                    )

                export_to_video(
                    video_with_force_prompt_aesthetic, filename_generated_video_with_force_prompt_aesthetic, fps=8
                )

        clear_objs_and_retain_memory([pipe])

        # del pipeline_args, videos, control_signal_video

    del pipe
    torch.cuda.empty_cache()
    import gc
    gc.collect()


