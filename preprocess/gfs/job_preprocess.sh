#!/bin/bash -l
#
# NOTE: This workflow is for generating extended domain for GFS forecast or detection
#       mode. Other mode of pre-training or finetuning will not need this script, but
#       directly run the script gfs_preprocess.py after editing the config.json
#
# INPUT: This script requires all GFS data to be ready under the ./input/gfs/detection
#       or ./input/gfs/forecast/YYYYMMDDHH properly.
#
# OUTPUT: A set of extended data extracted from the global GFS input, which will be
#       later used by the prediction step
#
# HIST: - 01, Oct 25: Created by CK
#
# AUTH: Chanh Kieu (Indiana University, Bloomington. Email: ckieu@iu.edu)
#
#==========================================================================

#SBATCH -A r00043
#SBATCH -J prediction
#SBATCH -t 09:59:00
#SBATCH -N 1
#SBATCH -p gpu 
#SBATCH --gpus 2
#SBATCH --mem-per-gpu=240G
##SBATCH --cpus-per-gpu=64
src="/N/slate/ckieu/tcg-net/"
cd "${src}/preprocess/gfs/"
yyyymmddhh=$1
run_mode=$2
if [ "${yyyymmddhh}" = "" ] || [ "${run_mode}" = "" ]; then
    echo "Must have 2 inputs for the data type and run mode to predict...exit 1"
    echo "E.g,: sh job_sbatch 2025100200 detection/forecast"
    exit 1
fi

check_data=`grep GFS_RAW ./config.json | grep ${run_mode} | grep ${yyyymmddhh}`
if [ "${check_data}" = "" ]; then
    echo "WARNING: This script is runing for forecast or detection mode only"
    echo "GFS_RAW in config.json will be now replaced for this mode"
    if [ "${run_mode}" = "forecast" ]; then
        sed -i '/GFS_RAW/c\        "GFS_RAW": "./input/gfs/forecast/'${yyyymmddhh}'"'  ${src}/config.json
    elif [ "${run_mode}" = "detection" ]; then
        sed -i '/GFS_RAW/c\        "GFS_RAW": "./input/gfs/detection/'${yyyymmddhh}'"' ${src}/config.json
    else
        echo "run mode ${run_mode} is not support... exit 2"
        exit 2
    fi
else
    check_detection=`grep GFS_RAW ./config.json | grep detection`
    check_forecast=`grep GFS_RAW ./config.json | grep forecast`
    if [ "${check_forecast}" != "" ]; then
        echo "GFS forecast data for cycle ${yyyymmddhh} is ready"
    elif [ "${check_detection}" != "" ]; then
        echo "GFS detection data for cycle ${yyyymmddhh} is ready"
    fi
fi

# rename GFS forecast files. This step is no longer critical, as the time 
# issue will be handled by the gfs_preprocess.py, using "valid_time". However, it will be kept
# here as the gfs_preprocess.py can only read file with extension "grib1", "grib2", and ".f000"
if [ "${run_mode}" = "detection" ]; then
    sh ./rename_file_datetime.sh ${yyyymmddhh} "${src}/input/gfs/${run_mode}"
fi

# generate the extend domain data
conda deactivate >& /dev/null
module load python/gpu/3.10.10
python gfs_preprocess.py
