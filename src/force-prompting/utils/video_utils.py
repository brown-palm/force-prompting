from typing import List, Optional, Tuple, Union
import torch
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.models.embeddings import get_3d_rotary_pos_embed
import subprocess, tempfile, cv2
import numpy as np
import os


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin

def encode_video(vae, accelerator, video):
    video = video.to(accelerator.device, dtype=vae.dtype)
    video = video.permute(0, 2, 1, 3, 4)  # [Batch, Channel, Frame, Height, Width]
    latent_dist = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor
    return latent_dist.permute(0, 2, 1, 3, 4).to(memory_format=torch.contiguous_format)

def tensor_to_video_ffmpeg(tensor, output_filename, fps=8):
    """
    Convert a PyTorch tensor to an MP4 video file using FFmpeg.
    
    Args:
        tensor (torch.Tensor): Tensor of shape (num_frames, channels, height, width)
                              with values in range [0, 1] or [0, 255]
        output_filename (str): Output filename ending with .mp4
        fps (int, optional): Frames per second. Defaults to 30.
    """
    # Check if tensor is on GPU and move to CPU if necessary
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Get tensor dimensions
    num_frames, channels, height, width = tensor.shape
    
    # Ensure tensor values are within [0, 255] range
    if tensor.max() <= 1.0:
        tensor = tensor * 255
    
    # Convert to numpy and ensure uint8 format
    frames = tensor.numpy().astype(np.uint8)
    
    # Create a temporary directory to store the frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save each frame as a PNG file
        for i in range(num_frames):
            if channels == 3:
                # Convert from (C, H, W) to (H, W, C)
                frame = frames[i].transpose(1, 2, 0)
                # Convert from RGB to BGR for OpenCV
                frame = frame[:, :, ::-1]
            elif channels == 1:
                frame = frames[i].squeeze(0)
                frame = np.stack([frame, frame, frame], axis=2)
            else:
                raise ValueError(f"Unsupported number of channels: {channels}. Expected 1 or 3.")
            
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, frame)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
        
        # Use FFmpeg to convert the frames to a video
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-profile:v', 'high',
            '-crf', '20',  # Quality factor (lower is better)
            '-pix_fmt', 'yuv420p',  # Standard pixel format for compatibility
            output_filename
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"Video saved to {output_filename}")
            print(f"Video dimensions: {width}x{height}, {num_frames} frames at {fps} FPS")
        except subprocess.CalledProcessError as e:
            print(f"Error creating video: {e}")
        except FileNotFoundError:
            print("FFmpeg is not installed or not in the PATH. Please install FFmpeg.")