#!/usr/bin/env python3
import os
import json
import subprocess
import glob
import argparse

def update_json_with_background(json_path, wind_speed, wind_angle, hdri_background, last_frame_path):
 
    if os.path.exists(json_path):
        # Load existing JSON
        with open(json_path, 'r') as file:
            data = json.load(file)
        # print(f"Loaded existing JSON with {len(data)} entries")
    else:
        # Create new dictionary
        data = {}
        # print("Created new JSON dictionary")

    hdri_background_name = hdri_background.split(".exr")[0]

    # CASE 1: if the background is not in the directory, then we add it
    if not hdri_background_name in data:
        data[hdri_background_name] = {
            "wind_speed"        : wind_speed,
            "wind_angle"        : wind_angle,
            "last_frame_path"   : last_frame_path
        }
    # CASE 2: if the background is in the directory, then we replace it if 
    # we like the current wind speed and angle better than that wind speed and angle
    elif hdri_background_name in data:
    
        # check if the wind speed is better
        speed_is_large_enough = wind_speed > 4000

        # check if the wind angle is better
        right_enough = (315 < wind_angle < 360) or (0 < wind_angle < 45)
        left_enough = 180 - 45 < wind_angle < 180 + 45
        angle_is_good_enough = right_enough or left_enough

        if speed_is_large_enough and angle_is_good_enough:
            data[hdri_background_name] = {
                "wind_speed"        : wind_speed,
                "wind_angle"        : wind_angle,
                "last_frame_path"   : last_frame_path
            }
    
    # Save the updated dictionary
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    return None

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
            
        wind_speed = params.get("wind_speed", 0)
        wind_angle = params.get("wind_angle", 0)
    except Exception as e:
        print(f"Error reading parameters file: {e}")
        return False
    
    json_path = os.path.join(frames_dir, "params.json")
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    if json_data['hdri_background'] == None:
        background_name = "none"
    else:
        background_name = json_data['hdri_background'].split(".exr")[0]
    
    # Format the output filename
    output_filename = f"flag_sample_{wind_speed:.1f}_0.0_{wind_angle:.1f}_0.0_background_{background_name}.mp4"
    output_dir = os.path.join(os.path.dirname(os.path.dirname(frames_dir)), "videos")
    os.makedirs(output_dir, exist_ok=True)
    
    # check if this one's background is already represented; if it's not, we need to add it to the master...
    # note: we always want to keep the one with the largest wind speed, and ideally a reasonable angle
    backgrounds_json_path = os.path.join(os.path.dirname(os.path.dirname(frames_dir)), "backgrounds.json")
    frame_names = os.listdir(frames_dir)
    frame_names.remove("params.json")
    last_frame_name = max(frame_names)
    last_frame_path = os.path.join(frames_dir, last_frame_name)
    # update_json_with_background(
    #     backgrounds_json_path, 
    #     json_data['wind_speed'],
    #     json_data['wind_angle'],
    #     json_data['hdri_background'],
    #     last_frame_path
    # )

    output_path = os.path.join(output_dir, output_filename)
    
    # Get a list of PNG files
    png_files = sorted(glob.glob(os.path.join(frames_dir, "frame*.png")))
    if not png_files:
        # print(f"Error: No PNG files found in {frames_dir}")
        return False
    NUM_PNGS_WE_NEED = 240
    if len(png_files) < NUM_PNGS_WE_NEED:
        # print(f"Error: Fewer than {NUM_PNGS_WE_NEED} found in {frames_dir}")
        return False
    if os.path.isfile(output_path):
        # print(f"Error: Skipping {frames_dir} because {output_path} already exists")
        return False
    
    # Determine the input pattern for ffmpeg
    frame_pattern = os.path.join(frames_dir, "frame%04d.png")
    
    # Build and execute the ffmpeg command
    try:
        cmd = [
            "ffmpeg", 
            "-y",  # Overwrite output file if it exists
            "-framerate", "24",
            "-start_number", "121",  # Start from frame 121
            "-i", frame_pattern,
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
        import sys, math
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