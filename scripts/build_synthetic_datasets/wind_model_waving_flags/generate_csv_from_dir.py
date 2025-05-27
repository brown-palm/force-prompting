import os
import argparse
import csv
import cv2
import json
from tqdm import tqdm
import numpy as np

def analyze_video(video_path, backgrounds_json):
    """Analyze a video file and extract basic information."""
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate duration in seconds
    seconds = frame_count / fps if fps > 0 else 0
    
    # Release the video capture object
    cap.release()

    # get wind speed and angle
    fname = os.path.basename(video_path) # e.g. 'flag_sample_100.2_0.0_7.5_0.0_background_garden_nook_4k.mp4'
    force_info = fname.split("_0.0_background_")[0] # e.g 'flag_sample_100.2_0.0_7.5'
    background_name = fname.split("_0.0_background_")[1].split(".mp4")[0] # e.g. 'garden_nook_4k'

    force_info_split = force_info.split("_")
    speed = float(force_info_split[2])
    angle = float(force_info_split[4])

    caption = backgrounds_json[background_name]['optimized_prompt']

    return {
        'video': os.path.basename(video_path),
        'caption': caption, 
        'frame': frame_count,
        'fps': fps,
        'seconds': seconds,
        'wind_speed': speed,
        'wind_angle': angle,
    }

def analyze_image(image_path):
    """Analyze a video file and extract basic information."""

    prompts_json = os.path.join(os.path.dirname(os.path.dirname(image_path)), "prompts_seed.json")
    with open(prompts_json) as f:
        prompts = json.load(f)

    image_name = os.path.basename(image_path).split(".png")[0]
    caption = prompts['optimized_prompts'][image_name]

    return {
        'image': os.path.basename(image_path),
        'caption': caption, 
        'wind_speed': 0.0,
        'wind_angle': 0.0,
    }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze video or image in a directory and output results to CSV.')
    parser.add_argument('--file_dir', help='Directory containing video or image files')
    parser.add_argument('--file_type', choices=["video", "image"])
    parser.add_argument('--output_path', help='Path to output CSV file')
    parser.add_argument('--backgrounds_json_path', help='Path to output CSV file')
    parser.add_argument('--subset_size', type=int, default=None, help='Optional subset size for video files')


    # Parse arguments
    args = parser.parse_args()

    # load backgrounds json
    with open(args.backgrounds_json_path, 'r') as f:
        backgrounds_json = json.load(f)
    
    # Check if file_dir exists
    if not os.path.isdir(args.file_dir):
        print(f"Error: Directory {args.file_dir} does not exist")
        return
    
    # Get all files
    if args.file_type == "video":
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    elif args.file_type == "image":
        extensions = ['png']

    files = []
    for file in os.listdir(args.file_dir):
        if any(file.lower().endswith(ext) for ext in extensions):
            files.append(os.path.join(args.file_dir, file))
    
    if not files:
        print(f"No {args.file_type} files found in {args.file_dir}")
        return
    
    files = sorted(files)
    if args.subset_size is not None and args.file_type == "video":
        np.random.shuffle(files)
    files = files[:args.subset_size]
    
    if args.file_type == "video":
        # Analyze files
        min_speed, max_speed = np.inf, -np.inf
        results = []
        for path in tqdm(files):
            # print(f"Analyzing {path}...")
            result = analyze_video(path, backgrounds_json)
            if result:
                min_speed = min(result["wind_speed"], min_speed)
                max_speed = max(result["wind_speed"], max_speed)
                results.append(result)

        results.sort(key=lambda x: x['wind_speed'])
        
        FILTER_THE_RESULTS = False # TODO: ADD TO WRITE UP...
        if FILTER_THE_RESULTS:
            RADIUS = 45
            results_filtered = []
            for result in results:

                wind_speed = result["wind_speed"]
                wind_angle = result["wind_angle"]

                condition_1 = 0 <= wind_angle <= RADIUS
                condition_2 = 360 - RADIUS <= wind_angle <= 360
                condition = condition_1 or condition_2

                if condition:
                    results_filtered.append(result)
            results = results_filtered

        # Write results to CSV
        with open(args.output_path, 'w', newline='') as csvfile:
            if args.file_type == "video":
                fieldnames = ['video', 'wind_speed', 'wind_angle', 'frame', 'fps', 'seconds', 'caption']
            elif args.file_type == "image":
                fieldnames = ['image', 'wind_speed', 'wind_angle', 'caption']
        
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:

                NORMALIZE_WIND_SPEED = True
                if NORMALIZE_WIND_SPEED:
                    result["wind_speed"] = (result["wind_speed"] - min_speed) / (max_speed - min_speed)
                    result["wind_speed"] = "{:.4f}".format(result["wind_speed"])

                writer.writerow(result)
    
        print(f"Analysis complete. Results written to {args.output_path}")

    # each image gets its own csv
    elif args.file_type == "image":
        # Make sure output directory exists for image CSVs
        output_dir = args.output_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Process each image file separately
        for path in files:
            print(f"Analyzing {path}...")
            result = analyze_image(path)
            
            # Create individual CSV for this image
            image_basename = os.path.basename(path)
            image_name = os.path.splitext(image_basename)[0]
            csv_path = os.path.join(output_dir, f"{image_name}.csv")
            
            # Write single result to CSV
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['image', 'wind_speed', 'wind_angle', 'caption']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                writer.writerow(result)
            
            print(f"Created CSV for {image_basename} at {csv_path}")
        
        print(f"Analysis complete. Individual CSVs written to {output_dir}")

if __name__ == "__main__":
    main()