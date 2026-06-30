#!/bin/bash -l
#SBATCH -A r00043
#SBATCH -J training
#SBATCH -t 05:59:00
#SBATCH -N 1
#SBATCH -p gpu
##SBATCH -p gpu-debug 
#SBATCH --gpus 2
#SBATCH --mem-per-gpu=240G
set -x
src="/N/slate/ckieu/tcg-net/"
output_dir="/N/scratch/ckieu/tcg-net/output"
cd "${src}/models/resnet18/"
step=$1
if [ "${step}" = "" ]; then
    echo "Must have an input for the step (leadtime) to resnet18...exit 1"
    exit 1
fi
echo "Training model for forecast lead time (step) = ${step}"
conda deactivate
module load python/gpu/3.10.10

# set up parameters
ratio=30
weight=6
project="ResNet"
data_source="MERRA2"
train_start_year=1955
train_end_year=2023
test_start_year=1940
test_end_year=1954
seed=45
batch_size=512

# link data source
rm -f ${src}/output/DynamicDomain ${src}/output/PastDomain ${src}/output/POSITIVE
ln -sf ${output_dir}/${data_source}_dynamic ${src}/output/DynamicDomain
ln -sf ${output_dir}/${data_source}_past ${src}/output/PastDomain
ln -sf $output_dir/${data_source}_positive ${src}/output/POSITIVE

# creating a csv for all data
python generate_csv_all.py --inp_dir "${src}/output/" --out_dir "${src}/output/csv/"

# splitting the csv for train/test/validation
python split_data_training.py \
    --inp_dir "${src}/output/csv/" \
    --out_dir "${src}/output/csv/" \
    --train_start_year ${train_start_year} \
    --train_end_year ${train_end_year} \
    --test_start_year ${test_start_year} \
    --test_end_year ${test_end_year} \
    --rus_ratio ${ratio}

# runing train job
python ./train_job.py \
    --project ${project} \
    --seed ${seed} \
    --ratio ${ratio} \
    --step ${step} \
    --weight ${weight} \
    --inp_dir "${src}/output/csv/DynamicRemain_rus${ratio}/Step_${step}" \
    --out_dir "${src}/models/pre-trained/dynamic/${project}_r${ratio}_w${weight}/Step_${step}" \
