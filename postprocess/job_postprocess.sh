#!/bin/bash -l
#
# NOTE: This script is to plot the map TCG prediction from a trained/finetuned model, which
#       are generated for 0-54 hr forecast lead time from the prediction step 
#
#       The script is used for plotting operational mode only. Not for pretrain/finetued mode
#
# INPUT: This script requires output from the prediction (inference) steps, which should exist under
#       /output/XXX_slide, where XXX is the global input data type (GFS/ERA5/IFS...)
#
# OUTPUT: a 2D map of 5x5 TCG probability that is contained from the file located under PREDICTION_CSV
#       in config.json
#
# HIST: - 29, Sep 25: Created by CK
#
# AUTH: Chanh Kieu (Indiana University, Bloomington. Email: ckieu@iu.edu)
#
#==========================================================================
#SBATCH -A r00043
#SBATCH -J postprocess
#SBATCH -t 00:59:00
#SBATCH -N 1
#SBATCH -p gpu-debug
#SBATCH --gpus 2
#SBATCH --mem-per-gpu=240G
##SBATCH --cpus-per-gpu=64
src="/N/slate/ckieu/tcg-net/"
cd "${src}/postprocess/"
yyyymmddhh=$1
data_type=$2
if [ "${data_type}" = "" ] || [ "${yyyymmddhh}" = "" ] ; then
    echo "Must have 2 inputs for the time and data type to predict...exit 1"
    echo "E.g.: sh job_sbatch 2025100300 gfs"
    exit 1
fi
echo "Plotting for data type: ${data_type} and time ${yyyymmddhh}"

conda deactivate >& /dev/null
module load python/gpu/3.10.10
ratio=30
weight=6   
project="ResNet"

# plot the TCG density map for all forecast lead times 
DATATYPE=${data_type^^}
sed -i '/PREDICT_CSV_FILE/c\        "PREDICT_CSV_FILE": "./output/'${DATATYPE}'_slide/dynamic_prediction.csv",'  ${src}/config.json
sed -i '/TCG_CYCLE/c\        "TCG_CYCLE": "'${yyyymmddhh}'"'  ${src}/config.json
python main.py

# plot the seasonality of the TCG frequency
#python plot_TCGfrequency_monthly.py
 
