#!/bin/bash -l
#
# NOTE: This script is to run the prediction (inference) from a trained/finetuned model
#       that will generate 0-54 hr forecast from each given WNP domain (output from 
#       the preprocess step).
# 
#       If it takes too much time to complete the slide step, need to run job_dynamics.sh
#       to finish up the second step (main.py).
#
# INPUT: This script requires output from the preprocess steps, which should exist under
#       /output/XXX_extend, where XXX is the global input data type (GFS/ERA5/IFS...)
#
# OUTPUT: a prediction file in the csv form, which is saved to the location PREDICTION_CSV.
#       Also, the intermediate output for each slide is saved under the variable DYNAMIC_DOMAIN
#       of the config.json
#       
# HIST: - 29, Sep 25: Created by CK
#
# AUTH: Chanh Kieu (Indiana University, Bloomington. Email: ckieu@iu.edu)
#
#==========================================================================
#SBATCH -A r00043
#SBATCH -J prediction
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu 
#SBATCH --gpus 2
#SBATCH --mem-per-gpu=240G
##SBATCH --cpus-per-gpu=64
src="/N/slate/ckieu/tcg-net/"
cd "${src}/models/prediction/"
data_type=$1
if [ "${data_type}" = "" ]; then
    echo "Must have an input for the data type to predict...exit 1"
    exit 1
fi
echo "Prediction using data type: ${data_type}"

conda deactivate >& /dev/null
module load python/gpu/3.10.10

# set up parameters
ratio=30
weight=6
pretrain_model="ResNet"
finetune_model="ResNet"
seed=45
#exp="_4060"
exp=""
model="${src}/models/finetune/${finetune_model}_r${ratio}_w${weight}"
if [ -d $model ]; then
    echo "Model ${model} exist... continue"
else
    echo "Model ${model} DOES NOT exist... exit 1"
    exit 1
fi

# runing map slide first
DATATYPE=${data_type^^}
sed -i '/INPUT_PATH/c\        "INPUT_PATH": "./output/'${DATATYPE}'_extend'${exp}'",'  ${src}/config.json
sed -i '/OUTPUT_PATH/c\        "OUTPUT_PATH": "./output/'${DATATYPE}'_slide"'  ${src}/config.json
sed -i '/SLIDE_DIR/c\        "SLIDE_DIR": "../../output/'${DATATYPE}'_slide/",'  ${src}/config.json
sed -i '/PREDICTION_CSV/c\        "PREDICTION_CSV": "../../output/'${DATATYPE}'_slide/dynamic_prediction.csv"'  ${src}/config.json
sed -i '/MODEL_TEMP/c\        "MODEL_TEMP": "'${model}'/Step_{0}_v0/checkpoints/last.ckpt",' ${src}/config.json
python map_slide.py --dataset ${data_type}

# now run prediction
python main.py
