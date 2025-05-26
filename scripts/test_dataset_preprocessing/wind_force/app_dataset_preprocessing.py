from flask import Flask, render_template, request, send_file, jsonify
import os
import io
import base64
import cv2
import json
import csv
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)

# Create necessary directories
output_dir = "datasets/wind-force/test/custom"
os.makedirs(output_dir, exist_ok=True)

from mimetypes import guess_type
def image_to_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop', methods=['POST'])
def crop_image():
    file = request.files['image']
    x = float(request.form['x'])
    y = float(request.form['y'])
    width = float(request.form['width'])
    height = float(request.form['height'])
    aspect_ratio = float(request.form['aspect_ratio'])
    
    # Get original filename for later use
    original_filename = file.filename
    
    # Open and crop the image
    img = Image.open(file.stream)
    cropped = img.crop((x, y, x + width, y + height))
    
    # Resize based on aspect ratio
    if abs(aspect_ratio - 1.5) < 0.01:
        target_size = (720, 480)  # Width x Height for 1.5 aspect ratio
    elif abs(aspect_ratio - 1.7708) < 0.01:
        target_size = (1360, 768)  # Width x Height for 1.7708 aspect ratio
    else:
        # Default - don't resize
        target_size = None
    
    # Apply resizing if target size is specified
    if target_size:
        resized = cropped.resize(target_size, Image.Resampling.LANCZOS)
    else:
        resized = cropped
    
    # Generate a new filename with underscore prefix
    base_name = os.path.basename(original_filename)
    new_filename = f"_{os.path.splitext(base_name)[0]}.png"
    file_dir = os.path.join(output_dir, "images")
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, new_filename)
    
    # Save the image to disk
    resized.save(file_path)
    
    # Save to bytes for direct response
    img_io = io.BytesIO()
    resized.save(img_io, 'PNG')
    img_io.seek(0)
    
    # Return both the file and its path
    response = send_file(img_io, mimetype='image/png')
    response.headers['X-Filename'] = new_filename
    response.headers['X-Filepath'] = file_path
    return response

@app.route('/optimize_prompt', methods=['POST'])
def optimize_prompt():
    data = request.json
    prompt = data.get('prompt', '')
    image_path = data.get('image_path', '')
    
    if not os.path.exists(image_path):
        return jsonify({'error': f'Image not found: {image_path}'}), 404
    
    # Use the image-to-video (i2v) prompt optimization logic
    try:
        client = OpenAI(api_key=api_key)
        
        sys_prompt_i2v = """
**Objective**: **Give a highly descriptive video caption based on input image and user input. **. As an expert, delve deep into the image with a discerning eye, leveraging rich creativity, meticulous thought. When describing the details of an image, include appropriate dynamic information to ensure that the video caption contains reasonable actions and plots. If user input is not empty, then the caption should be expanded according to the user's input. 

**Note**: The input image is the first frame of the video, and the output video caption should describe the motion starting from the current image. User input is optional and can be empty. 

**Note**: Don't contain camera transitions!!! Don't contain screen switching!!! Don't contain perspective shifts !!!

**Answering Style**:
Answers should be comprehensive, conversational, and use complete sentences. The answer should be in English no matter what the user's input is. Provide context where necessary and maintain a certain tone.  Begin directly without introductory phrases like "The image/video showcases" "The photo captures" and more. For example, say "A woman is on a beach", instead of "A woman is depicted in the image".

**Output Format**: "[highly descriptive image caption here]"

user input:
"""     
        # Call the API with image and prompt
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"{sys_prompt_i2v}"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_to_url(image_path),
                            },
                        },
                    ],
                },
            ],
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=250,
        )
        
        optimized_prompt = response.choices[0].message.content
        return jsonify({'optimized_prompt': optimized_prompt})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Removed the get_pixel route as we're no longer selecting pixels

@app.route('/write_csv', methods=['POST'])
def write_csv():
    data = request.json
    image_path = data.get('image_path')
    caption = data.get('caption')
    wind_angle = data.get('wind_angle', 0.0)
    wind_speed = data.get('wind_speed', 0.5)
    prev_prompt_counter = data.get('prev_prompt_counter', 0)
    
    # Get image dimensions
    img = Image.open(image_path)
    width, height = img.size
    
    # Determine the prompt counter - only incrementing prompt counter now
    prompt_counter = prev_prompt_counter + 1
    
    # Generate CSV name based on image name
    image_basename = os.path.basename(image_path)
    image_name = os.path.splitext(image_basename)[0]
    csv_filename = f"{image_name}_prompt{prompt_counter}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Create the result dictionary
    result = {
        'image': image_basename,
        'caption': caption,
        'width': width,
        'height': height,
        'wind_angle': float(wind_angle),
        'wind_speed': float(wind_speed)
    }
    
    # Write the CSV
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image', 'wind_angle', 'wind_speed', 'width', 'height', 'caption']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerow(result)
    
    return jsonify({
        'success': True,
        'csv_path': csv_path,
        'prompt_counter': prompt_counter
    })

if __name__ == '__main__':
    app.run(debug=True)