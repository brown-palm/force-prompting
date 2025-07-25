import base64
from openai import OpenAI
import datetime
import subprocess
import time
import sys
import os

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
    NUM_SEEDS = 5
    INPUT_IMAGE_PATH = "temp/wheelbarrow1.png"
    prompt = "Remove all the dirt from this wheelbarrow, so it'll be empty Keep everything else the same, including the sizes and appearances and style of all the objects"
    
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