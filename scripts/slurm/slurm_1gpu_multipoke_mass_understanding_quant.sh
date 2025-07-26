#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --constraint=a6000|l40s|geforce3090
#SBATCH --exclude=gpu1506,gpu2108,gpu2109,gpu2112,gpu2113,gpu2114,gpu2115,gpu2116
#SBATCH -N 1 # gives one node, makes sure cpu cores are on same node
#SBATCH -c 1 # num CPU cores
#SBATCH --mem=24G
#SBATCH -t 3:00:00
#SBATCH -e output/slurm_logs/%A_%a.err
#SBATCH -o output/slurm_logs/%A_%a.out
#SBATCH --mail-user=nate_gillman@brown.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-10

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

# python scripts/print_csv_paths.py datasets/point-force/test/benchmark/apple

###########################################################################
###########################################################################
###########################################################################
################# OUR FINAL BENCHMARK FOR WIND FORCE ######################
###########################################################################
###########################################################################
###########################################################################
declare -a IMAGE_CSVS=(
    "datasets/point-force/test/benchmark_multipoke_mass_understanding_quant/___massexpsoccervsbowling01.csv"
    "datasets/point-force/test/benchmark_multipoke_mass_understanding_quant/___massexpsoccervsbowling02.csv"
    "datasets/point-force/test/benchmark_multipoke_mass_understanding_quant/___massexpsoccervsbowling03.csv"
    "datasets/point-force/test/benchmark_multipoke_mass_understanding_quant/___massexpsoccervsbowling04.csv"
    "datasets/point-force/test/benchmark_multipoke_mass_understanding_quant/___massexpsoccervsbowling05.csv"
    "datasets/point-force/test/benchmark_multipoke_mass_understanding_quant/___massexpsoccervsbowling06.csv"
    "datasets/point-force/test/benchmark_multipoke_mass_understanding_quant/___massexpsoccervsbowling07.csv"
    "datasets/point-force/test/benchmark_multipoke_mass_understanding_quant/___massexpsoccervsbowling08.csv"
    "datasets/point-force/test/benchmark_multipoke_mass_understanding_quant/___massexpsoccervsbowling09.csv"
    "datasets/point-force/test/benchmark_multipoke_mass_understanding_quant/___massexpsoccervsbowling10.csv"
    "datasets/point-force/test/benchmark_multipoke_mass_understanding_quant/___massexpsoccervsbowling11.csv"
)

# set the checkpoint path here
CHECKPOINT="checkpoints/step-5000-checkpoint-point-force-copy.pt"

# Get the current job's CSV file using the SLURM_ARRAY_TASK_ID environment variable
CURRENT_CSV=${IMAGE_CSVS[$SLURM_ARRAY_TASK_ID]}
echo "Processing file: $CURRENT_CSV"

for image_csv in "${IMAGE_CSVS[@]}"; do
  bash scripts/inference_1_gpu.sh \
      --force_type "point_force" \
      --model_type "controlnet_with_force_control_signal" \
      --num_validation_videos 1 \
      --csv_path_val ${CURRENT_CSV} \
      --pretrained_controlnet_path "${CHECKPOINT}"
done