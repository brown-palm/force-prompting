#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --constraint=a6000|l40s|geforce3090
#SBATCH --exclude=gpu1506,gpu2108,gpu2109,gpu2112,gpu2113,gpu2114,gpu2115,gpu2116
#SBATCH -N 1 # gives one node, makes sure cpu cores are on same node
#SBATCH -c 1 # num CPU cores
#SBATCH --mem=24G
#SBATCH -t 2:00:00
#SBATCH -e output/slurm_logs/%j.err
#SBATCH -o output/slurm_logs/%j.out
#SBATCH --mail-user=nate_gillman@brown.edu
#SBATCH --mail-type=ALL

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
IDXS=("datasets/wind-force/test/benchmark/bubbles/_bubbles1_prompt1.csv") 
IDXS=("datasets/wind-force/test/benchmark/campfire/_campfire2_benchmark.csv" "datasets/wind-force/test/benchmark/campfire/_campfire4.csv") 
IDXS=("datasets/wind-force/test/benchmark/chimney/_chimney1.csv" "datasets/wind-force/test/benchmark/chimney/_chimney2_benchmark.csv") 
IDXS=("datasets/wind-force/test/benchmark/clothwithperson/_clothwithperson1.csv") 
IDXS=("datasets/wind-force/test/benchmark/confetti/_confetti1_prompt1.csv" "datasets/wind-force/test/benchmark/confetti/_confetti1_prompt2.csv" "datasets/wind-force/test/benchmark/confetti/_confetti2_prompt1.csv" "datasets/wind-force/test/benchmark/confetti/_confetti2_prompt2.csv") 
IDXS=("datasets/wind-force/test/benchmark/dress/_dress1_benchmark.csv" "datasets/wind-force/test/benchmark/dress/_dress3_benchmark.csv") 
IDXS=("datasets/wind-force/test/benchmark/fallingleaves/_fallingleaves1_prompt1.csv" "datasets/wind-force/test/benchmark/fallingleaves/_fallingleaves1_prompt2.csv" "datasets/wind-force/test/benchmark/fallingleaves/_fallingleaves2_prompt1.csv" "datasets/wind-force/test/benchmark/fallingleaves/_fallingleaves2_prompt2.csv" "datasets/wind-force/test/benchmark/fallingleaves/_fallingleaves4_prompt1.csv" "datasets/wind-force/test/benchmark/fallingleaves/_fallingleaves4_prompt2.csv") 
IDXS=("datasets/wind-force/test/benchmark/fog/_fog1_prompt1.csv" "datasets/wind-force/test/benchmark/fog/_fog2_prompt1.csv" "datasets/wind-force/test/benchmark/fog/_fog2_prompt2.csv" "datasets/wind-force/test/benchmark/fog/_fog3_prompt1.csv" "datasets/wind-force/test/benchmark/fog/_fog3_prompt2.csv" "datasets/wind-force/test/benchmark/fog/_fog4_prompt2.csv") 
IDXS=("datasets/wind-force/test/benchmark/hair/_hair1_vary_angles_benchmark.csv") 
IDXS=("datasets/wind-force/test/benchmark/inflatabletube/_inflatabletube3_prompt1.csv" "datasets/wind-force/test/benchmark/inflatabletube/_inflatabletube3_prompt2.csv") 
IDXS=("datasets/wind-force/test/benchmark/litter/_litter1_prompt1.csv" "datasets/wind-force/test/benchmark/litter/_litter1_prompt2.csv") 
IDXS=("datasets/wind-force/test/benchmark/paperlantern/_paperlantern1_prompt1.csv" "datasets/wind-force/test/benchmark/paperlantern/_paperlantern1_prompt2.csv" "datasets/wind-force/test/benchmark/paperlantern/_paperlantern3_prompt1.csv" "datasets/wind-force/test/benchmark/paperlantern/_paperlantern3_prompt2.csv") 
IDXS=("datasets/wind-force/test/benchmark/smokeincense/_smokeincense1_prompt1.csv") 
IDXS=("datasets/wind-force/test/benchmark/snow/_snow1_prompt1.csv" "datasets/wind-force/test/benchmark/snow/_snow1_prompt2.csv" "datasets/wind-force/test/benchmark/snow/_snow2_prompt1.csv" "datasets/wind-force/test/benchmark/snow/_snow2_prompt2.csv") 
IDXS=("datasets/wind-force/test/benchmark/steamybeverage/_steamybeverage2_prompt1.csv" "datasets/wind-force/test/benchmark/steamybeverage/_steamybeverage2_prompt2.csv") 
IDXS=("datasets/wind-force/test/benchmark/whitecloth/_whitecloth1_benchmark.csv" "datasets/wind-force/test/benchmark/whitecloth/_whitecloth2_benchmark.csv" "datasets/wind-force/test/benchmark/whitecloth/_whitecloth3.csv" "datasets/wind-force/test/benchmark/whitecloth/_whitecloth4.csv")
# all of them concatenated
IMAGE_CSVS=()


