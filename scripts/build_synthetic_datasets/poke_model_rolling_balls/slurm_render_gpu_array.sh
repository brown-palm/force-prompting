#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --constraint=a6000|l40s|geforce3090
#SBATCH --exclude=gpu1506,gpu2108,gpu2112,gpu2114,gpu2115,gpu2116
#SBATCH -N 1 # num nodes
#SBATCH -c 1 # num CPU cores
#SBATCH --mem=14G
#SBATCH --array=0-120
#SBATCH -t 12:00:00
#SBATCH -e output/slurm_logs/%j.err
#SBATCH -o output/slurm_logs/%j.out

# SET UP COMPUTING ENV
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Activate virtual environment
# Load anaconda module, and other modules
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
CONDA_ENV_DIR=/oscar/data/superlab/users/nates_stuff/cogvideox-controlnet-clean/conda-env
conda activate $CONDA_ENV_DIR

# Move to correct working directory
HOME_DIR=/oscar/data/superlab/users/nates_stuff/cogvideox-controlnet-clean
cd ${HOME_DIR}

# experiment script here
bash scripts/build_synthetic_datasets/poke_model_rolling_balls/rolling_balls_render.sh