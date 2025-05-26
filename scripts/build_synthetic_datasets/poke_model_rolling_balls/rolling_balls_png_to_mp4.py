#!/usr/bin/env python3
import os
import json
import subprocess
import glob
import argparse
import math
import sys

def create_mp4_from_frames(frames_dir):
    """
    Creates an MP4 file from PNG frames in the given directory.
    Uses the parameters.json file to name the output file.
    """
    # Check if the directory exists
    if not os.path.isdir(frames_dir):
        # print(f"Error: Directory {frames_dir} does not exist")
        return False
    
    # Check if params.json exists
    params_file = os.path.join(frames_dir, "params.json")
    if not os.path.exists(params_file):
        # print(f"Error: Parameters file {params_file} not found")
        return False
    
    # Load parameters
    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
        force_strength = params.get('GENERATED_FORCE_STRENGTH', 0)
        force_angle = params.get('PIXEL_ANGLE', 0)
        force_location_x = params.get('BALL_PIXEL_COORDS', [-1,-1])[0]
        force_location_y = params.get('BALL_PIXEL_COORDS', [-1,-1])[1]
        background = params.get('GROUND_TEXTURE', "uh_oh")

        is_bowling_ball = params.get("MOVING_BALL_NAME") == "bowling_ball"
        is_bowling_ball_str = "_bowling" if is_bowling_ball else ""
    except Exception as e:
        print(f"Error reading parameters file: {e}")
        return False

    # Format the output filename
    output_filename = f"background_{background}_angle_{force_angle:.4f}_force_{force_strength:.4f}_coordx_{force_location_x}_coordy_{force_location_y}{is_bowling_ball_str}.mp4"
    output_dir = os.path.join(os.path.dirname(os.path.dirname(frames_dir)), "videos")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_filename)
    if os.path.isfile(output_path):
        # print(f"Error: Skipping {frames_dir} because {output_path} already exists")
        return False
    
    # Get a list of PNG files
    png_files = sorted(glob.glob(os.path.join(frames_dir, "frame*.png")))
    if not png_files:
        # print(f"Error: No PNG files found in {frames_dir}")
        return False
    NUM_PNGS_WE_NEED = 130
    if len(png_files) < NUM_PNGS_WE_NEED:
        # print(f"Error: Fewer than {NUM_PNGS_WE_NEED} found in {frames_dir}")
        return False

    
    # Determine the input pattern for ffmpeg
    frame_pattern = os.path.join(frames_dir, "frame%04d.png")
    
    # Build and execute the ffmpeg command
    try:
        START_NUMBER = 11
        cmd = [
            "ffmpeg", 
            "-y",  # Overwrite output file if it exists
            "-framerate", "24",
            "-start_number", f"{START_NUMBER}",  # Start from frame 121
            "-i", frame_pattern,
            "-frames:v", "120",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"Successfully created {output_path}")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
        return False

def process_all_render_dirs(base_dir):
    """
    Find all directories that contain rendered frames and process them.
    """
    # Find directories that contain params.json files
    processed_count = 0
    error_count = 0

    png_dirs = sorted(os.listdir(base_dir))

    try:
        SLURM_ARRAY_TASK_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
        NUM_SLURM_JOBS = 100
        LOWER_IDX_FOR_THIS_JOB = math.floor(SLURM_ARRAY_TASK_ID * len(png_dirs) / NUM_SLURM_JOBS)
        UPPER_IDX_FOR_THIS_JOB = math.floor((SLURM_ARRAY_TASK_ID+1) * len(png_dirs) / NUM_SLURM_JOBS)
        png_dirs = png_dirs[LOWER_IDX_FOR_THIS_JOB:UPPER_IDX_FOR_THIS_JOB]
        print("SLURM_ARRAY_TASK_ID = ", SLURM_ARRAY_TASK_ID)
        sys.stdout.flush()
    except:
        pass
    
    for dir in png_dirs:
        # print(f"Processing directory: {root}")
        png_dir = os.path.join(base_dir, dir)
        if create_mp4_from_frames(png_dir):
            processed_count += 1
        else:
            error_count += 1
    
    print(f"Processing complete: {processed_count} mp4s created, {error_count} directories skipped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create MP4s from rendered Blender frames")
    parser.add_argument("base_dir", nargs="?", default=None, 
                        help="Base directory to search for rendered frames")
    parser.add_argument("--dir", "-d", help="Process a specific render directory")
    
    args = parser.parse_args()
    
    if args.base_dir:
        process_all_render_dirs(args.base_dir)
    else:
        print("Error: Please specify either a base directory or a specific render directory")
        parser.print_help()