import base64
from openai import OpenAI
import datetime
import subprocess
import time
import sys
import os
from dotenv import load_dotenv # <-- ADD THIS
load_dotenv()

def run_single_edit(input_path, prompt, index):
    """Create a single image edit as a subprocess"""
    # Create a temporary script file that will be executed as a subprocess
    temp_script = f"temp_script_{index}.py"
    
    random_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[-15:]
    output_path = input_path.split(".png")[0] + f"_{random_str}.png"
    
    with open(temp_script, "w") as f:
        f.write(f'''
import base64
from openai import OpenAI
import datetime

client = OpenAI()

INPUT_IMAGE_PATH = "{input_path}"
prompt = "{prompt}"

random_str = "{random_str}"
OUTPUT_IMAGE_PATH = "{output_path}"

PROMPT = f"""
{{prompt}}
"""

result = client.images.edit(
    model="gpt-image-1",
    image=open(INPUT_IMAGE_PATH, "rb"),
    prompt=PROMPT
)

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# Save the image to a file
with open(OUTPUT_IMAGE_PATH, "wb") as f:
    f.write(image_bytes)

print(f"Image saved to {{OUTPUT_IMAGE_PATH}}")
''')
    
    # Run the temporary script as a subprocess
    subprocess.Popen([sys.executable, temp_script])
    
    # Return the temp script name for cleanup later
    return temp_script

def main():
    # Configuration
    NUM_SEEDS = 10
    INPUT_IMAGE_PATH = (
        "datasets/point-force/test/mass_understanding_quantitative/wood/images/_materialballrollingballonwoodsoccer2_11:16:00.595702.png"
    )
    # prompt = "Edit this photo as follows. Zoom out, have there be two ornaments (one is this one, and one is a a wooden version of this one), and they should be on opposite sides of the frame." #so that there are two laundry baskets, one which is empty on the top left side of the frame, and one which is full on the top right corner of the frame. Make sure there is nothing else in the frame. Make it a top-down view, and make sure that they're at the same height."
    # prompt = "Edit this photo so that there are two skateboards, one with a brick on them, and the other empty. The skateboards should be identical, with the only difference being that one has a brick on it. Put one of them in the bottom left of the screen, and the other in the bottom right of the screen. They should be parallel and facing upwards. I.e. I need them to be oriented so that the side of the skateboard is parallel with the side of the frame."
    # prompt = "Rotate the skateboard so that they're oriented vertically, i.e. I need the side of it to be parallel with the side of the frame."
    # prompt = "Edit this photo as follows. Zoom out, make it a top down view that's slightly angled from the side, and have a stack of books in the bottom left corner, and have a single book in the bottom right corner. They should be on opposite sides of the frame, and the same number of pixels from the bottom of the frame." #so that there are two laundry baskets, one which is empty on the top left side of the frame, and one which is full on the top right corner of the frame. Make sure there is nothing else in the frame. Make it a top-down view, and make sure that they're at the same height."
    # prompt += " Keep the photo's aspect ratio the same (landscape, a wide photo), and make sure it's very zoomed out, so that the objects are only a small fraction of the screen."
    # prompt += "Keep everything else the same, including the sizes and appearances and style of all the objects"

    # prompt = "Edit this image by adding it so that there is a bowling ball on the left side of the frame. Don't change the soccer ball that is already there, and make sure the bowling ball is the same exact size as the soccer ball."
    prompt = "Edit this image by making the left bowling ball smaller so that it is the same size as the right soccer ball. Also make sure that the soccer and bowling ball are at the same height, i.e. that they're the same distance from the bottom of the frame."

    pause_seconds = 0.01
    
    temp_scripts = []
    
    # Launch all processes with a small pause between them
    for i in range(NUM_SEEDS):
        temp_script = run_single_edit(INPUT_IMAGE_PATH, prompt, i)
        temp_scripts.append(temp_script)
        time.sleep(pause_seconds)
        print(f"Started process {i+1} of {NUM_SEEDS}")
    
    print(f"All {NUM_SEEDS} processes have been launched.")
    
    # Optional: You can add code here to clean up the temporary scripts after some time
    # This example waits 10 seconds then removes the temporary scripts
    time.sleep(0.05)  
    for script in temp_scripts:
        try:
            os.remove(script)
            print(f"Removed temporary script: {script}")
        except:
            pass

if __name__ == "__main__":
    main()