#!/bin/bash -l
#SBATCH -A r00043
#SBATCH -J preprocess
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus 2
#SBATCH --mem-per-gpu=240G
##SBATCH --cpus-per-gpu=64
set -x
src="/N/slate/ckieu/tcg-net/"
data_type=$1
cd "${src}/preprocess/${data_type}"
if [ "${data_type}" = "" ]; then
    echo "Must have an input for the type of global input (era5/merra2/gfs/ncep/cmip6)...exit 1"
    exit 1
fi
echo "Generating extend domain for ${data_type}"

conda deactivate
module load python/gpu/3.10.10
python ${data_type}_preprocess.py
