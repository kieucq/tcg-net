#!/bin/bash -l
#
# NOTE: This script is to run the main step only, after the slide step is done. It is 
#       needed because the slide step in job_prediction.sh takes more time than allowed.
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
#SBATCH -J dynamics
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu 
#SBATCH --gpus 2
#SBATCH --mem-per-gpu=240G
##SBATCH --cpus-per-gpu=64
src="/N/slate/ckieu/tcg-net/"
cd "${src}/models/prediction/"
conda deactivate >& /dev/null
module load python/gpu/3.10.10
python main.py
