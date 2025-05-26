import os
import argparse
import csv
import cv2
import json
from tqdm import tqdm
import numpy as np
import random

EXTREMAL_FORCES = {
    "balls" : {
        "min" : 8.0,
        "max" : 64.0,
    },
    "plants" : {
        "min" : 0.1,
        "max" : 5.0,
    },
}

def analyze_video(video_path, backgrounds_json, type):
    """Analyze a video file and extract basic information."""
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate duration in seconds
    seconds = frame_count / fps if fps > 0 else 0
    
    # Release the video capture object
    cap.release()

    # get wind speed and angle
    fname = os.path.basename(video_path)
    fname = fname.split(".mp4")[0].split("_") # e.g. ['flag', 'sample', '12', '0.0', '0.021163130408621503', '90.0']

    angle = float(fname[fname.index("angle")+1])
    coordx = int(fname[fname.index("coordx")+1])
    coordy = int(fname[fname.index("coordy")+1])

    force = float(fname[fname.index("force")+1])
    force_normalized = (force - EXTREMAL_FORCES[type]["min"]) / (EXTREMAL_FORCES[type]["max"] - EXTREMAL_FORCES[type]["min"])

    if type == "balls":
        background = os.path.basename(video_path).split("background_")[1].split("_angle")[0]
        caption = backgrounds_json[background]["optimized_prompt"]
    elif type == "plants":
        videos_json_path = video_path.replace(".mp4", ".json")
        with open(videos_json_path) as f:
            videos_json = json.load(f)
        caption = videos_json["caption"]
        angle = np.clip(angle + 0.4 * np.random.randn(), a_min=0.0, a_max=360.0)
    else:
        raise NotImplementedError

    return {
        'video': os.path.basename(video_path),
        'caption': caption, 
        'frame': frame_count,
        'fps': fps,
        'seconds': seconds,
        'width': width,
        'height': height,
        'angle': angle,
        'force': force_normalized,
        'coordx': coordx,
        'coordy': coordy,
    }

def analyze_image(image_path):
    """Analyze an image file and extract basic information."""
    # Read the image to get dimensions
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not open image {image_path}")
        return None
    
    # Get image dimensions
    height, width = img.shape[:2]

    prompts_json = os.path.join(os.path.dirname(os.path.dirname(image_path)), "prompts_seed.json")
    with open(prompts_json) as f:
        prompts = json.load(f)

    image_name = os.path.basename(image_path).split(".png")[0]
    caption = prompts['optimized_prompts'][image_name]

    return {
        'image': os.path.basename(image_path),
        'caption': caption, 
        'width': width,
        'height': height,
        'angle': 0.0,
        'force': 40.0,
        'coordx': 0,
        'coordy': 0,
    }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze video or image in a directory and output results to CSV.')
    parser.add_argument('--file_dir', help='Directory containing video or image files')
    parser.add_argument('--file_type', choices=["video", "image"])
    parser.add_argument('--output_path', help='Path to output CSV file')
    parser.add_argument('--backgrounds_json_path_soccer')
    parser.add_argument('--backgrounds_json_path_bowling')
    parser.add_argument('--take_subset_size', default=-1, type=int)

    # Parse arguments
    args = parser.parse_args()

    # load backgrounds json
    try:
        with open(args.backgrounds_json_path_soccer, 'r') as f:
            backgrounds_json_soccer = json.load(f)
        with open(args.backgrounds_json_path_bowling, 'r') as f:
            backgrounds_json_bowling = json.load(f)
    except:
        print("At least one of the backgrounds jsons is missing. Assuming you're only trying to make a plant csv.")
    
    # Check if file_dir exists
    if not os.path.isdir(args.file_dir):
        print(f"Error: Directory {args.file_dir} does not exist")
        return
    
    # Get all files
    if args.file_type == "video":
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    elif args.file_type == "image":
        extensions = ['png']

    files_balls, files_plants = [], []
    for file in os.listdir(args.file_dir):
        if any(file.lower().endswith(ext) for ext in extensions):
            if file.startswith("background"):
                files_balls.append(os.path.join(args.file_dir, file))
            elif file.startswith("carnation"):
                files_plants.append(os.path.join(args.file_dir, file))

    if not files_balls and not files_plants:
        print(f"No {args.file_type} files found in {args.file_dir}")
        return
    
    files_balls = sorted(files_balls)
    files_plants = sorted(files_plants)
    
    if args.file_type == "video":

        # Analyze ball videos
        results_balls = []
        if files_balls:
            if args.take_subset_size > -1:
                files_balls = random.sample(files_balls, args.take_subset_size)
            for path in tqdm(files_balls):
                if "bowling.mp4" in path:
                    result = analyze_video(path, backgrounds_json_bowling, type="balls")
                else:
                    result = analyze_video(path, backgrounds_json_soccer, type="balls")
                if result:
                    results_balls.append(result)
            results_balls.sort(key=lambda x: x['angle'])

        # Analyze plant videos
        results_plants = []
        if files_plants:
            if args.take_subset_size > -1:
                files_plants = random.sample(files_plants, args.take_subset_size)
            for path in tqdm(files_plants):
                result = analyze_video(path, None, type="plants")
                if result:
                    results_plants.append(result)
            results_plants.sort(key=lambda x: x['angle'])

        results = results_balls + results_plants

        # Write results to CSV
        with open(args.output_path, 'w', newline='') as csvfile:
            fieldnames = ['video', 'angle', 'force', 'coordx', 'coordy', 'frame', 'fps', 'seconds', 
                         'width', 'height', 'caption']
        
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                # Format angle and force to 2 decimal places before writing
                result['angle'] = format(result['angle'], '.4f')
                result['force'] = format(result['force'], '.4f')
                writer.writerow(result)
    
        print(f"Analysis complete. Results written to {args.output_path}")


if __name__ == "__main__":
    main()