#!/bin/bash -l
#
# NOTE: This workflow is for running the entire system in the operation mode, with 
#       2 options of "forecast" or "detection" using operational global input from
#       GFS or IFS (not yet checked).
#
# INPUT: This script requires the 1) input cycle, 2) run mode, and 3) input data type 
#
#       Note that this script will check for all GFS/IFS data under the 
#       ./input/gfs/detection/YYYYMMDDHH or ./input/gfs/forecast/YYYYMMDDHH.
#
# OUTPUT: A set of 2D map of TCG density for each forecast lead time, starting from
#       input cycle, which is saved under ./output/postprocess/YYYYMMDDHH 
#
# HIST: - 04, Oct 25: Created by CK
#
# AUTH: Chanh Kieu (Indiana University, Bloomington. Email: ckieu@iu.edu)
#
#==========================================================================
#SBATCH -A r00043
#SBATCH -J operation
#SBATCH -t 08:01:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus 2
#SBATCH --mem-per-gpu=240G
src="/N/slate/ckieu/tcg-net/"
cd "${src}/operation/"

# getting input parameters and cross check. run_mode must be equal to detection or forecast
yyyymmddhh=$1
run_mode=$2
data_type=$3
if [ "${yyyymmddhh}" = "" ] || [ "${run_mode}" = "" ] || [ "${data_type}" = "" ]; then
    echo "Must have 3 inputs for the cycle, data type, run mode ...exit 1"
    echo "E.g,: sh job_main 2025100200 detection (forecast) gfs"
    exit 1
fi

# check if global input data exists before proceeding
check_file=`ls ${src}/input/${data_type}/${run_mode}/${yyyymmddhh}/${data_type}*pgrb2*.f*`
check_grib2=`ls ${src}/input/${data_type}/${run_mode}/${yyyymmddhh}/${data_type}*.grib2`
if [ "${check_file}" = "" ] && [ "${check_grib2}" = "" ]; then
    echo "Data ${data_type} with run mode ${run_mode} for cycle ${yyyymmddhh} is not ready"
    exit 1
fi

# running preprocess. This must go to a specific directory corresponding to that data type
echo "Running the preprocess step for ${data_type}, ${run_mode}, ${yyyymmddhh} ..."
cd ${src}/preprocess/${data_type}
sh job_preprocess.sh ${yyyymmddhh} ${run_mode}

# running prediction step
echo "Running the prediction step for ${data_type}, ${run_mode}, ${yyyymmddhh} ..."
cd ${src}/models/prediction/
sh job_prediction.sh ${data_type}

# running postprocess step
echo "Producing TCG density map for ${data_type}, ${run_mode}, ${yyyymmddhh} ..."
cd ${src}/postprocess/
sh job_postprocess.sh ${yyyymmddhh} ${data_type}

# cleaning up and transfer data to different locations
