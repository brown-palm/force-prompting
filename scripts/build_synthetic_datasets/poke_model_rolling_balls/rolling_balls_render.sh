#####################################################################################
#####################################################################################
#################### FOR USER TO CONFIGURE: things in this block ####################
#####################################################################################
#####################################################################################
module load blender/4.4.0-446jdgt       # need blender
RENDER_DIR_ROOT=~/scratch/rolling_balls # set to render directory
#####################################################################################
#####################################################################################

# Set up render directory
RENDER_DIR=${RENDER_DIR_ROOT}/pngs
mkdir -p $RENDER_DIR

# Path to blender file
SOURCE_BLEND="scripts/build_synthetic_datasets/poke_model_rolling_balls/rolling_balls.blend"

# Run Blender
echo "Starting Blender render with copied file..."
for i in {1..1000}
do
    blender \
        --enable-autoexec \
        -b "${SOURCE_BLEND}" \
        -P scripts/build_synthetic_datasets/poke_model_rolling_balls/rolling_balls.py \
        -s 1 \
        -e 130 \
        -x 1 \
        -o ${RENDER_DIR}/frame#### \
        -a
done
