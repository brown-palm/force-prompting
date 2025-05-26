<div align="center">

# Force Prompting: Video Generation Models Can<br>Learn and Generalize Physics-based Control Signals

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-<COLOR>.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project page](https://img.shields.io/badge/-Project%20page-blue.svg)](https://force-prompting.github.io/)

</div>

The official PyTorch implementation of the paper [**"Force Prompting: Video Generation Models Can<br>Learn and Generalize Physics-based Control Signals"**](https://arxiv.org/abs/XXXX.XXXXX).
Please visit our [**webpage**](https://force-prompting.github.io/) for more details.


## First time setup instructions


<details>
  <summary><b> Create conda environment </b></summary>

<br>

This has been tested on: `Driver Version: 535.129.03   CUDA Version: 12.2`.

Create conda environment:
```bash
CONDA_ENV_DIR=${PWD}/conda-env
conda create -p $CONDA_ENV_DIR python=3.11
conda activate $CONDA_ENV_DIR

# install torch
pip install --prefix=$CONDA_ENV_DIR torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
# it's a good idea to check that the torch installation was successful
python -c 'import torch; print(torch.cuda.is_available()); a = torch.zeros(5); a = a.to("cuda:0"); print(a)'

# install all the other requirements
pip install -r requirements.txt --prefix=$CONDA_ENV_DIR 
```

</details>








## Running inference using our checkpoints



<details>
  <summary><b> (Optional) preprocess your own data </b></summary>
<br>

If you want to run inference on either the point force model or the wind force model, and you want to do it on your own images, then we recommend using the flask app that we built for data preprocessing. 
This app provides a unified UI which takes care of details like generating a CSV with the relevant contents, taking a screenshot of the image into the correct resolution and aspect ratio, selecting force magnitude and direction, and putting things into appropriate folders, and upscaling the prompt.
And if you're using the point force model, allowing you to select the pixel coordinates to poke.

In order to use the prompt upscaling part of our Flask app, you will need an OpenAI API key; we recommend creating a `.env` file and adding the line `OPENAI_API_KEY=<your_key>`.


<details>
  <summary><i>More details in case you're curious:</i> </summary>
<br>

* As our models are built on top of CogVideoX, the ideal input image resolution is 720x480. 
* Additionally, you must specify a detailed text prompt during generation (this is due to us using CogVideoX as our base model; for more details, check out their paper). 
For example, prompts like *"the flower moves"* don't work as well as detailed prompts such as *"A lone dandelion stands tall against the backdrop of a vibrant sunset, its delicate seeds illuminated by the warm glow. The dandelion sways gracefully back and forth, its fluffy seeds trembling slightly with each movement. The sky transitions from deep blue to a fiery orange, casting a serene and magical atmosphere over the scene. The surrounding grass whispers softly, adding to the tranquil symphony of nature as the day slowly fades into night."*
Note that you'll need an OpenAI API key for prompt upscaling, unless you want to write your own very detailed prompts.
Technically the model will be able to run without this prompt upscaling step, but the results are likely to be worse because it would be out of domain for CogVideoX.
* If you want to skip this prompt upscaling step because you don't have an API key, your options are to 1) use a ChatGPT/Claude/etc web app to upscale for free, using the prompt in [scripts/test_dataset_preprocessing/point_force/app_dataset_preprocessing.py](scripts/test_dataset_preprocessing/point_force/app_dataset_preprocessing.py); or 2) type your prompt directly into the "upscaled prompt" box in the Flask app.
* For the point force model, the pixel coordinates are expected to be values between $720$ (horizontally) and $480$ (vertically).
Our convention is that the lower left pixel value is $(0,0)$ and the upper right pixel value is $(719, 479)$.
* For both models, the force magnitude is normalized to between $[0,1]$, and the force angle accepts degree values in the interval $[0,360)$, with $0$ indicating a force to the right, $90$ indicating upwards, etc.
</details>

**Tip:** If you're running this on a server using VSCode, then port forwarding will happen automatically and the flask app will work as intended. However, you can avoid latency issues by running locally—if you’re preprocessing a lot of data you may find the latency burdensome.<br>

**Tip:** The force prompting models tend to do well at modeling physical phenomena that the base CogVideoX model can already do well at (e.g. swaying plants) and tends to do worse on things CogVideoX doesn't do so well at (e.g. collisions).
If you find that the force prompting model doesn't do well on a given example, you should consider training a new Force Prompting model on a video generative model with a stronger physics prior and let us know how it goes :)


### Dataset preprocessing flask app, point force model

The following flask app will output csvs to `datasets/point-force/test/custom/*.csv` and their corresponding images to `datasets/point-force/test/custom/images/*.png`.
To run inferece on this csv, just use this path for the generated CSV in the inference script below.

**Tip:** Make sure there are no spaces in the file name for the image that you upload to the Flask app—only letters, numbers, dashes, or underscores.

```bash
python scripts/test_dataset_preprocessing/point_force/app_dataset_preprocessing.py
```



### Dataset preprocessing flask app, wind force model

The following flask app will output csvs to `datasets/wind-force/test/custom/*.csv` and their corresponding images to `datasets/wind-force/test/custom/images/*.png`.
To run inferece on this csv, just use this path for the generated CSV in the inference script below.

```bash
python scripts/test_dataset_preprocessing/wind_force/app_dataset_preprocessing.py
```








</details>


<details>
  <summary><b> Download model checkpoints </b></summary>

<br>

If you want to run inference using either of the pretrained models, then running the following script will download both checkpoints.

```bash
python scripts/download_files/download_checkpoints.py
```

If download was successful, the checkpoints should be organized like this:

```
checkpoints/
├── step-5000-checkpoint-point-force.pt
└── step-5000-checkpoint-wind-force.pt
```




</details>



<details>
  <summary><b> Point force model: inference </b></summary>

<br>

Running the following script will generate videos using your chosen checkpoint and image/text/force prompt. **This script will output videos into the same directory as the input checkpoint.** For example, if you use the checkpoint `checkpoints/step-5000-checkpoint-point-force.pt`, then the videos will be output into the directory `checkpoints/step-5000-checkpoint-point-force/`.

```bash
# this is our pretrained model; you can change to your own path
CHECKPOINT="checkpoints/step-5000-checkpoint-point-force.pt"

# you can change this to the list of csvs you want to run inference on.
IMAGE_CSVS=(
  "datasets/point-force/test/mass_understanding_quantitative/wood/_materialballrollingballonwoodbowling1_obj1_prompt1.csv"
)

for image_csv in "${IMAGE_CSVS[@]}"; do
  bash scripts/inference_1_gpu.sh \
      --force_type "point_force" \
      --model_type "controlnet_with_force_control_signal" \
      --num_validation_videos 1 \
      --csv_path_val "${image_csv}" \
      --pretrained_controlnet_path "${CHECKPOINT}"
done
```

If you want to run inference on some preprocessed data, you can find the `IMAGE_CSVS` inside the directory [datasets/point-force/test/benchmark/](datasets/point-force/test/benchmark/).
This directory contains our benchmark test dataset, plus additional images and prompt configurations.
The list of configurations for just our benchmark dataset can be found at [datasets/poke-force/test/benchmark/benchmark_details.csv](datasets/point-force/test/benchmark/benchmark_details.csv).


</details>





<details>
  <summary><b> Wind force model: inference </b></summary>

<br>

Running the following script will generate videos using your chosen checkpoint and image/text/force prompt. **This script will output videos into the same directory as the input checkpoint.** For example, if you use the checkpoint `checkpoints/step-5000-checkpoint-wind-force.pt`, then the videos will be output into the directory `checkpoints/step-5000-checkpoint-wind-force/`.

```bash
# this is our pretrained model; you can change to your own path
CHECKPOINT="checkpoints/step-5000-checkpoint-wind-force.pt"

# you can change this to the list of csvs you want to run inference on.
IMAGE_CSVS=(
  "datasets/wind-force/test/benchmark/bubbles/_bubbles1_prompt1.csv"
)

for image_csv in "${IMAGE_CSVS[@]}"; do
  bash scripts/inference_1_gpu.sh \
      --force_type "wind_force" \
      --model_type "controlnet_with_force_control_signal" \
      --num_validation_videos 1 \
      --csv_path_val "${image_csv}" \
      --pretrained_controlnet_path "${CHECKPOINT}"
done
```

If you want to run inference on some preprocessed data, you can find the `IMAGE_CSVS` inside the directory [datasets/wind-force/test/benchmark/](datasets/wind-force/test/benchmark/).
This directory contains our benchmark test dataset, plus additional images and prompt configurations.
The list of configurations for just our benchmark dataset can be found at [datasets/wind-force/test/benchmark/benchmark_details.csv](datasets/wind-force/test/benchmark/benchmark_details.csv).



</details>


## Training the force prompting model

<details>
  <summary><b> Download training datasets </b></summary>

<br>


If you want to train either model from scratch, then the following script will download all of our training data.

```bash
python scripts/download_files/download_datasets.py
```

If the download was successful, then the datasets should be organized like this:

```
datasets/
├── point-force/
│   └── train/
│       ├── point_force_23000/
│       │   ├── background_aerial_beach_01_4k_angle_0.4076_force_17.8989_coordx_159_coordy_407_bowling.mp4
│       │   ├── background_aerial_beach_01_4k_angle_0.4889_force_25.3516_coordx_448_coordy_186_bowling.mp4
│       │   └── ...
│       └── point_force_23000.csv
└── wind-force/
    └── train/
        ├── wind_force_15359/
        │   ├── flag_sample_0.1_0.0_321.3_0.0_background_qwantani_dusk_2_4k.mp4
        │   ├── flag_sample_1.8_0.0_77.4_0.0_background_golden_gate_hills_4k.mp4
        │   └── ...
        └── wind_force_15359.csv
```




</details>

<details>
  <summary><b> Point force model: training </b></summary>


### Train from scratch

```bash
bash scripts/train_4_gpu.sh \
    --force_type "point_force" \
    --video_root_dir "datasets/point-force/train/point_force_23000" \
    --csv_path "datasets/point-force/train/point_force_23000.csv"
```

### Resume training from checkpoint

```bash
# replace with your checkpoint path
RESUME_FROM_CHECKPOINT="output/point_force/2025-05-08_03-46-47/step-4500-checkpoint.pt" 

bash scripts/train_4_gpu.sh \
    --force_type "point_force" \
    --video_root_dir "datasets/point-force/train/point_force_23000" \
    --csv_path "datasets/point-force/train/point_force_23000.csv" \
    --pretrained_controlnet_path $RESUME_FROM_CHECKPOINT
```


</details>



<details>
  <summary><b> Wind force model: training </b></summary>

  ### Train from scratch

```bash
bash scripts/train_4_gpu.sh \
    --force_type "wind_force" \
    --video_root_dir "datasets/wind-force/train/wind_force_15359" \
    --csv_path "datasets/wind-force/train/wind_force_15359.csv"
```

### Resume training from checkpoint


```bash
# replace with your checkpoint path
RESUME_FROM_CHECKPOINT="output/wind_force/2025-05-18_21-46-01/step-2000-checkpoint.pt" 

bash scripts/train_4_gpu.sh \
    --force_type "wind_force" \
    --video_root_dir "datasets/wind-force/train/wind_force_15359" \
    --csv_path "datasets/wind-force/train/wind_force_15359.csv" \
    --pretrained_controlnet_path $RESUME_FROM_CHECKPOINT
```


</details>



## Generate training datasets (in progress!)

<details>
  <summary><b> Download Blender assets </b></summary>

<br>

If you want to generate synthetic training data yourself for either task, then you'll need to run the following script, which will download all of the assets that Blender needs to generate a diverse dataset.

```bash
python scripts/download_files/download_blender_textures.py
```

If the download was successful, then the datasets should be organized like this:

```
.cache/
├── football_textures/
│   ├── 1/
│   ├── 2/
│   └── …  
├── ground_textures/
│   ├── aerial_beach_01_4k.blend/
│   ├── aerial_grass_rock_4k.blend/
│   └── …  
└── HDRIs/
    ├── acoustical_shell_4k.exr
    ├── air_museum_playground_4k.exr
    └── …  
```

</details>





<details>
  <summary><b> Point force model: generate Blender ball rolling videos, and Physdreamer plant swaying videos </b></summary>

<br>

These command line blender rendering scripts were tested on Blender 4.4.

### Step 1: Generate ball rolling videos using Blender

This script renders video frames to pngs.

```bash
sh scripts/build_synthetic_datasets/poke_model_rolling_balls/rolling_balls_render.sh
```

And this script concatenates the pngs to mp4s.

```bash
RENDER_DIR=~/scratch/rolling_balls/pngs
python scripts/build_synthetic_datasets/poke_model_rolling_balls/rolling_balls_png_to_mp4.py $RENDER_DIR
```

But I have a separate script for parallelizing these.


### Step 2: Generate plant swaying videos using PhysDreamer


We used the [PhysDreamer](https://github.com/a1600012888/PhysDreamer) repo for our codebase.
Our main modifications to their codebase allowed us to generate data at scale.
We plan to release those scripts soon.


### Step 3: Create the csv for the training data


We're assuming that we already have a directory of videos of soccer balls moving around, and another dir of videos of plants moving around.
The goal is to preprocess both of them to create a joint dataset.

```bash
# this dir is already filled with mp4s
DIR_BALLS="/users/ngillman/scratch/rolling_balls/videos"
# this dir is already filled with mp4s and jsons
DIR_PLANTS="/oscar/data/superlab/users/nates_stuff/cogvideox-controlnet/data/2025-04-07-point-force-unified-model/videos-05-11-ablation-no-bowling-balls-temp-justflowers"
# we eventually want to create this
DIR_COMBINED="datasets/point-force/train/point_force_23000_05-09"
```

We'll make a CSV for each of our temp directories.

```bash
# make balls csv
python scripts/build_synthetic_datasets/poke_model_rolling_balls/generate_csv_for_plants_and_balls_from_dir.py \
    --file_dir ${DIR_BALLS} \
    --file_type video \
    --output_path ${DIR_COMBINED}_balls.csv \
    --backgrounds_json_path_soccer scripts/build_synthetic_datasets/poke_model_rolling_balls/backgrounds_soccer.json \
    --backgrounds_json_path_bowling scripts/build_synthetic_datasets/poke_model_rolling_balls/backgrounds_bowling.json \
    --take_subset_size 12000

# make plants csv
python scripts/build_synthetic_datasets/poke_model_rolling_balls/generate_csv_for_plants_and_balls_from_dir.py \
    --file_dir ${DIR_PLANTS} \
    --file_type video \
    --output_path ${DIR_COMBINED}_plants.csv \
    --take_subset_size 11000
```

Combine the two csvs into one csv.

```bash
EXP_DIR=2025-04-07-point-force-unified-model
python scripts/build_synthetic_datasets/poke_model_rolling_balls/concatenate_csvs.py \
    --input_path_csv1 ${DIR_COMBINED}_balls.csv \
    --input_path_csv2 ${DIR_COMBINED}_plants.csv \
    --output_path_csv ${DIR_COMBINED}.csv
```

And finally copy all the mp4s into the combined directory.

```bash
mkdir -p ${DIR_COMBINED}
cp ${DIR_BALLS}/*.mp4 ${DIR_COMBINED}
cp ${DIR_PLANTS}/*.mp4 ${DIR_COMBINED}
```


</details>


<details>
  <summary><b> Wind force model: generate Blender flag waving videos</b></summary>



### Step 1: Generate flag waving videos using Blender

We plan to release this code soon.

### Step 2: Create the csv for the training data

We plan to release this code soon.

</details>




## Acknowledgments

We thank the authors of the works we build upon:
- [CogVideoX](https://github.com/THUDM/CogVideo)
- [TheDenk/cogvideox-controlnet](https://github.com/TheDenk/cogvideox-controlnet)
- [PhysDreamer](https://github.com/a1600012888/PhysDreamer)

## Bibtex

If you find this code useful in your research, please cite:

```
@InProceedings{TODO:addcitation
}
```

