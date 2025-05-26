#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --constraint=a6000|l40s|geforce3090
#SBATCH --exclude=gpu1506,gpu2108,gpu2109,gpu2112,gpu2113,gpu2114,gpu2115,gpu2116
#SBATCH -N 1 # gives one node, makes sure cpu cores are on same node
#SBATCH -c 1 # num CPU cores
#SBATCH --mem=24G
#SBATCH -t 14:00:00
#SBATCH -e output/slurm_logs/%A_%a.err
#SBATCH -o output/slurm_logs/%A_%a.out
#SBATCH --mail-user=nate_gillman@brown.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-23

# SET UP COMPUTING ENV
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Activate virtual environment
# Load anaconda module, and other modules
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES=0 # e.g. 0 or 0,1,2,3
module load cuda/12.2.0-4lgnkrh
CONDA_ENV_DIR=/oscar/data/superlab/users/nates_stuff/cogvideox-controlnet-clean/conda-env
conda activate $CONDA_ENV_DIR
HOME_DIR=/oscar/data/superlab/users/nates_stuff/cogvideox-controlnet-clean
cd ${HOME_DIR}


### Benchmark for material understanding, quantitative study
### 24 total
declare -a IMAGE_CSVS=(
    "datasets/point-force/test/mass_understanding_quantitative/dirt/_materialballrollingballondirtbowling1_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/dirt/_materialballrollingballondirtbowling2_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/dirt/_materialballrollingballondirtbowling3_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/dirt/_materialballrollingballondirtsoccer1_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/dirt/_materialballrollingballondirtsoccer2_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/dirt/_materialballrollingballondirtsoccer3_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/grass/_materialballrollingballongrassbowling1_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/grass/_materialballrollingballongrassbowling2_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/grass/_materialballrollingballongrassbowling3_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/grass/_materialballrollingballongrasssoccer1_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/grass/_materialballrollingballongrasssoccer2_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/grass/_materialballrollingballongrasssoccer3_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/stone/_materialballrollingballonstonebowling1_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/stone/_materialballrollingballonstonebowling2_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/stone/_materialballrollingballonstonebowling3_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/stone/_materialballrollingballonstonesoccer1_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/stone/_materialballrollingballonstonesoccer2_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/stone/_materialballrollingballonstonesoccer3_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/wood/_materialballrollingballonwoodbowling1_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/wood/_materialballrollingballonwoodbowling2_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/wood/_materialballrollingballonwoodbowling3_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/wood/_materialballrollingballonwoodsoccer1_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/wood/_materialballrollingballonwoodsoccer2_obj1_prompt1.csv" 
    "datasets/point-force/test/mass_understanding_quantitative/wood/_materialballrollingballonwoodsoccer3_obj1_prompt1.csv"
)

# Get the current job's CSV file using the SLURM_ARRAY_TASK_ID environment variable
CURRENT_CSV=${IMAGE_CSVS[$SLURM_ARRAY_TASK_ID]}
echo "Processing file: $CURRENT_CSV"

#######################################################
#######################################################
#######################################################
##### inference using a controlnet-trained model ######
#######################################################
#######################################################
#######################################################
CHECKPOINT="output/point_force/2025-05-08_17-27-30/step-5000-checkpoint.pt"
EXP_DIR=2025-04-07-point-force-unified-model
MODEL_TYPE="controlnet_with_force_control_signal"

bash scripts/point_force_inference_1_gpu.sh \
    --model_type "${MODEL_TYPE}" \
    --num_validation_videos 8 \
    --csv_path_val "${CURRENT_CSV}" \
    --pretrained_controlnet_path "${CHECKPOINT}"

