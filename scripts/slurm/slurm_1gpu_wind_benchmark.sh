#!/bin/bash

# A batch script for running a job on Oscar's 3090 condo, using the Slurm scheduler
# The 3090 condo runs NVIDIA's GeForce RTX 3090 graphics card

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --constraint=a6000|l40s|geforce3090
#SBATCH --exclude=gpu1506,gpu2108,gpu2109,gpu2112,gpu2113,gpu2114,gpu2115,gpu2116
#SBATCH -N 1 # gives one node, makes sure cpu cores are on same node
#SBATCH -c 1 # num CPU cores
#SBATCH --mem=24G
#SBATCH -t 1:00:00
#SBATCH -e output/slurm_logs/%A_%a.err
#SBATCH -o output/slurm_logs/%A_%a.out
#SBATCH --mail-user=nate_gillman@brown.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-89

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
################# OUR FINAL BENCHMARK FOR WIND FORCE; 44 of them ######################
###########################################################################
###########################################################################
###########################################################################
declare -a IMAGE_CSVS=(
    "datasets/wind-force/test/benchmark/bubbles/_bubbles1_prompt1.csv"
    "datasets/wind-force/test/benchmark/campfire/_campfire2_benchmark.csv"
    "datasets/wind-force/test/benchmark/campfire/_campfire4.csv"
    "datasets/wind-force/test/benchmark/chimney/_chimney1.csv" 
    "datasets/wind-force/test/benchmark/chimney/_chimney2_benchmark.csv"
    "datasets/wind-force/test/benchmark/clothwithperson/_clothwithperson1.csv"
    "datasets/wind-force/test/benchmark/confetti/_confetti1_prompt1.csv" 
    "datasets/wind-force/test/benchmark/confetti/_confetti1_prompt2.csv" 
    "datasets/wind-force/test/benchmark/confetti/_confetti2_prompt1.csv" 
    "datasets/wind-force/test/benchmark/confetti/_confetti2_prompt2.csv"
    "datasets/wind-force/test/benchmark/dress/_dress1_benchmark.csv" 
    "datasets/wind-force/test/benchmark/dress/_dress3_benchmark.csv"
    "datasets/wind-force/test/benchmark/fallingleaves/_fallingleaves1_prompt1.csv" 
    "datasets/wind-force/test/benchmark/fallingleaves/_fallingleaves1_prompt2.csv" 
    "datasets/wind-force/test/benchmark/fallingleaves/_fallingleaves2_prompt1.csv" 
    "datasets/wind-force/test/benchmark/fallingleaves/_fallingleaves2_prompt2.csv" 
    "datasets/wind-force/test/benchmark/fallingleaves/_fallingleaves4_prompt1.csv" 
    "datasets/wind-force/test/benchmark/fallingleaves/_fallingleaves4_prompt2.csv"
    "datasets/wind-force/test/benchmark/fog/_fog1_prompt1.csv" 
    "datasets/wind-force/test/benchmark/fog/_fog2_prompt1.csv" 
    "datasets/wind-force/test/benchmark/fog/_fog2_prompt2.csv" 
    "datasets/wind-force/test/benchmark/fog/_fog3_prompt1.csv" 
    "datasets/wind-force/test/benchmark/fog/_fog3_prompt2.csv"
    "datasets/wind-force/test/benchmark/fog/_fog4_prompt2.csv"
    "datasets/wind-force/test/benchmark/hair/_hair1_vary_angles_benchmark.csv"
    "datasets/wind-force/test/benchmark/inflatabletube/_inflatabletube3_prompt1.csv" 
    "datasets/wind-force/test/benchmark/inflatabletube/_inflatabletube3_prompt2.csv"
    "datasets/wind-force/test/benchmark/litter/_litter1_prompt1.csv" 
    "datasets/wind-force/test/benchmark/litter/_litter1_prompt2.csv"
    "datasets/wind-force/test/benchmark/paperlantern/_paperlantern1_prompt1.csv" 
    "datasets/wind-force/test/benchmark/paperlantern/_paperlantern1_prompt2.csv" 
    "datasets/wind-force/test/benchmark/paperlantern/_paperlantern3_prompt1.csv" 
    "datasets/wind-force/test/benchmark/paperlantern/_paperlantern3_prompt2.csv"
    "datasets/wind-force/test/benchmark/smokeincense/_smokeincense1_prompt1.csv"
    "datasets/wind-force/test/benchmark/snow/_snow1_prompt1.csv" 
    "datasets/wind-force/test/benchmark/snow/_snow1_prompt2.csv" 
    "datasets/wind-force/test/benchmark/snow/_snow2_prompt1.csv" 
    "datasets/wind-force/test/benchmark/snow/_snow2_prompt2.csv"
    "datasets/wind-force/test/benchmark/steamybeverage/_steamybeverage2_prompt1.csv" 
    "datasets/wind-force/test/benchmark/steamybeverage/_steamybeverage2_prompt2.csv"
    "datasets/wind-force/test/benchmark/whitecloth/_whitecloth1_benchmark.csv" 
    "datasets/wind-force/test/benchmark/whitecloth/_whitecloth2_benchmark.csv" 
    "datasets/wind-force/test/benchmark/whitecloth/_whitecloth3.csv" 
    "datasets/wind-force/test/benchmark/whitecloth/_whitecloth4.csv"
)

###########################################################################
###########################################################################
###########################################################################
################# OUR FINAL BENCHMARK FOR POINT FORCE; 90 of them #####################
###########################################################################
###########################################################################
###########################################################################
declare -a IMAGE_CSVSS=(
    "datasets/point-force/test/benchmark/apple/_apple1_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/apple/_apple1_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/apple/_apple2_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/apple/_apple2_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/apple/_apple3_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/apple/_apple3_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/apple/_apple4_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/apple/_apple4_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/balloon/_balloon3_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/balloon/_balloon3_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/balloon/_balloon4_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/balloon/_balloon4_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/blueberrybush/_blueberrybush1_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/blueberrybush/_blueberrybush1_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/blueberrybush/_blueberrybush2_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/blueberrybush/_blueberrybush2_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/blueberrybush/_blueberrybush3_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/blueberrybush/_blueberrybush3_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/blueberrybush/_blueberrybush4_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/blueberrybush/_blueberrybush4_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/dandelion/_dandelion1_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/dandelion/_dandelion1_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/dandelion/_dandelion3_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/dandelion/_dandelion3_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/dandelion/_dandelion4_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/dandelion/_dandelion4_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/ivy/_ivy1_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/ornament/_ornament1_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/ornament/_ornament1_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/ornament/_ornament2_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/ornament/_ornament2_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/ornament/_ornament4_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/rose/_rose2_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/rose/_rose2_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/rose/_rose2_obj1_prompt3.csv" 
    "datasets/point-force/test/benchmark/rose/_rose2_obj1_prompt4.csv" 
    "datasets/point-force/test/benchmark/rose/_rose2_obj1_prompt5.csv" 
    "datasets/point-force/test/benchmark/rose/_rose2_obj1_prompt6.csv" 
    "datasets/point-force/test/benchmark/rose/_rose3_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/rose/_rose3_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/rose/_rose3_obj1_prompt3.csv" 
    "datasets/point-force/test/benchmark/rose/_rose3_obj1_prompt4.csv" 
    "datasets/point-force/test/benchmark/rose/_rose3_obj1_prompt5.csv" 
    "datasets/point-force/test/benchmark/rose/_rose3_obj1_prompt6.csv" 
    "datasets/point-force/test/benchmark/rose/_rose4_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/rose/_rose4_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/rose/_rose4_obj1_prompt3.csv" 
    "datasets/point-force/test/benchmark/rose/_rose4_obj1_prompt4.csv" 
    "datasets/point-force/test/benchmark/rose/_rose4_obj1_prompt5.csv" 
    "datasets/point-force/test/benchmark/rose/_rose4_obj1_prompt6.csv" 
    "datasets/point-force/test/benchmark/rose/_rose5_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/rose/_rose5_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/rose/_rose5_obj1_prompt3.csv" 
    "datasets/point-force/test/benchmark/rose/_rose5_obj1_prompt4.csv" 
    "datasets/point-force/test/benchmark/rose/_rose5_obj2_prompt1.csv" 
    "datasets/point-force/test/benchmark/rose/_rose5_obj2_prompt2.csv" 
    "datasets/point-force/test/benchmark/rose/_rose5_obj2_prompt3.csv" 
    "datasets/point-force/test/benchmark/rose/_rose5_obj2_prompt4.csv" 
    "datasets/point-force/test/benchmark/sunflower/_sunflower2_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/sunflower/_sunflower2_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/sunflower/_sunflower3_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/sunflower/_sunflower3_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/swing/_swing3_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/swing/_swing3_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/toycar/_toycar1_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/toycar/_toycar2_obj1_prompt1_vary_speeds.csv" 
    "datasets/point-force/test/benchmark/toycar/_toycar2_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/toycar/_toycar3_obj1_prompt1_vary_speeds.csv" 
    "datasets/point-force/test/benchmark/toycar/_toycar3_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/toycar/_toycar4_obj1_prompt1_vary_speeds.csv" 
    "datasets/point-force/test/benchmark/toycar/_toycar4_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack1_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack1_obj2_prompt1.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack2_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack2_obj2_prompt1.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack2_obj3_prompt1.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack3_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack3_obj2_prompt1.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack3_obj3_prompt1.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack5_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack5_obj2_prompt1.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack6_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack6_obj1_prompt2.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack6_obj2_prompt1.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack6_obj2_prompt2.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack7_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack7_obj2_prompt1.csv" 
    "datasets/point-force/test/benchmark/toytrainontracks/_toytrainontrack7_obj3_prompt1.csv" 
    "datasets/point-force/test/benchmark/windmill/_windmill2_obj1_prompt1.csv" 
    "datasets/point-force/test/benchmark/windmill/_windmill2_obj1_prompt2.csv"
)

# set the checkpoint path here
# CHECKPOINT="output/wind_force/2025-07-24_15-30-21-wind-0.5x-size/step-5000-checkpoint.pt"
# CHECKPOINT="output/wind_force/2025-07-25_07-36-57-wind-2.0x-size/step-5000-checkpoint.pt"
CHECKPOINT="output/unified_point_and_wind_force/2025-07-26_22-48-58-unified/step-5000-checkpoint.pt"

# Get the current job's CSV file using the SLURM_ARRAY_TASK_ID environment variable
CURRENT_CSV=${IMAGE_CSVSS[$SLURM_ARRAY_TASK_ID]}
echo "Processing file: $CURRENT_CSV"

bash scripts/inference_1_gpu.sh \
    --force_type "point_force" \
    --model_type "controlnet_with_force_control_signal" \
    --num_validation_videos 1 \
    --csv_path_val ${CURRENT_CSV} \
    --pretrained_controlnet_path "${CHECKPOINT}"